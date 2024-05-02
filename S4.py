# This is heavily based on S4 model from the paper "Efficiently Modeling Long Sequences with Structured State Spaces"
# and uses (adjusting slightly) the implementation at "https://github.com/TariqAHassan/S4Torch/blob/main/s4torch/model.py"

from typing import Union, Optional, Type

import numpy as np
import torch
from torch import nn
from torch import view_as_real as as_real
from torch.fft import ifft, irfft, rfft
from torch.nn import functional as F
from torch.nn import init
from S4reqs import TemporalAdapter, Residual, SequentialWithResidual

def _log_step_initializer(
    tensor: torch.Tensor,  # values should be from U(0, 1)
    dt_min: float = 0.001,
    dt_max: float = 0.1,
) -> torch.Tensor:
    # Initializes the log step parameter
    scale = np.log(dt_max) - np.log(dt_min)
    return tensor * scale + np.log(dt_min)


def _make_omega_l(l_max: int, dtype: torch.dtype = torch.complex64) -> torch.Tensor:
    #Omega buffer used to obtain K
    return torch.arange(l_max).type(dtype).mul(2j * np.pi / l_max).exp()


def _make_hippo(N: int) -> np.ndarray:
    # Make Hippo matrix, which is used to compute the roots of the polynomial
    def idx2value(n: int, k: int) -> Union[int, float]:
        if n > k:
            return np.sqrt(2 * n + 1) * np.sqrt(2 * k + 1)
        elif n == k:
            return n + 1
        else:
            return 0

    hippo = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            hippo[i, j] = idx2value(i + 1, j + 1)
    return hippo


def _make_nplr_hippo(N: int) -> tuple[np.ndarray, ...]:
    # Make NPLR Hippo matrix
    nhippo = -1 * _make_hippo(N)

    p = 0.5 * np.sqrt(2 * np.arange(1, N + 1) + 1.0)
    q = 2 * p
    S = nhippo + p[:, np.newaxis] * q[np.newaxis, :]

    lambda_, V = np.linalg.eig(S)
    return lambda_, p, q, V


def _make_p_q_lambda(n: int) -> list[torch.Tensor]:
    # Make p, q, and lambda
    lambda_, p, q, V = _make_nplr_hippo(n)
    Vc = V.conj().T
    p = Vc @ p
    q = Vc @ q.conj()
    return [torch.from_numpy(i) for i in (p, q, lambda_)]


def _cauchy_dot(v: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    # Calculates the cauchy dot product
    if v.ndim == 1:
        v = v.unsqueeze(0).unsqueeze(0)
    elif v.ndim == 2:
        v = v.unsqueeze(1)
    elif v.ndim != 3:
        raise IndexError(f"Expected `v` to be 1D, 2D or 3D, got {v.ndim}D")
    return (v / denominator).sum(dim=-1)


def _non_circular_convolution(u: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    # Non-circular convolution
    l_max = u.shape[1]
    ud = rfft(F.pad(u.float(), pad=(0, 0, 0, l_max, 0, 0)), dim=1)
    Kd = rfft(F.pad(K.float(), pad=(0, l_max)), dim=-1)
    return irfft(ud.transpose(-2, -1) * Kd)[..., :l_max].transpose(-2, -1).type_as(u)


class S4Layer(nn.Module):
    """S4 Layer.

    Structured State Space for (Long) Sequences (S4) layer.

    Args:
        d_model (int): number of internal features
        n (int): dimensionality of the state representation
        l_max (int): length of input signal

    Attributes:
        omega_l (torch.Tensor): omega buffer (of length ``l_max``) used to obtain ``K``.
        ifft_order (torch.Tensor): (re)ordering for output of ``torch.fft.ifft()``.

    """

    omega_l: torch.Tensor
    ifft_order: torch.Tensor

    def __init__(self, d_model: int, n: int, l_max: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.l_max = l_max

        p, q, lambda_ = map(lambda t: t.type(torch.complex64), _make_p_q_lambda(n))
        self._p = nn.Parameter(as_real(p))
        self._q = nn.Parameter(as_real(q))
        self._lambda_ = nn.Parameter(as_real(lambda_).unsqueeze(0).unsqueeze(1))

        self.register_buffer(
            "omega_l",
            tensor=_make_omega_l(self.l_max, dtype=torch.complex64),
        )
        self.register_buffer(
            "ifft_order",
            tensor=torch.as_tensor(
                [i if i == 0 else self.l_max - i for i in range(self.l_max)],
                dtype=torch.long,
            ),
        )

        self._B = nn.Parameter(
            as_real(init.xavier_normal_(torch.empty(d_model, n, dtype=torch.complex64)))
        )
        self._Ct = nn.Parameter(
            as_real(init.xavier_normal_(torch.empty(d_model, n, dtype=torch.complex64)))
        )
        self.D = nn.Parameter(torch.ones(1, 1, d_model))
        self.log_step = nn.Parameter(_log_step_initializer(torch.rand(d_model)))

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, n={self.n}, l_max={self.l_max}"

    @property
    def p(self) -> torch.Tensor:
        return torch.view_as_complex(self._p)

    @property
    def q(self) -> torch.Tensor:
        return torch.view_as_complex(self._q)

    @property
    def lambda_(self) -> torch.Tensor:
        return torch.view_as_complex(self._lambda_)

    @property
    def B(self) -> torch.Tensor:
        return torch.view_as_complex(self._B)

    @property
    def Ct(self) -> torch.Tensor:
        return torch.view_as_complex(self._Ct)

    def _compute_roots(self) -> torch.Tensor:
        a0, a1 = self.Ct.conj(), self.q.conj()
        b0, b1 = self.B, self.p
        step = self.log_step.exp()

        g = torch.outer(2.0 / step, (1.0 - self.omega_l) / (1.0 + self.omega_l))
        c = 2.0 / (1.0 + self.omega_l)
        cauchy_dot_denominator = g.unsqueeze(-1) - self.lambda_

        k00 = _cauchy_dot(a0 * b0, denominator=cauchy_dot_denominator)
        k01 = _cauchy_dot(a0 * b1, denominator=cauchy_dot_denominator)
        k10 = _cauchy_dot(a1 * b0, denominator=cauchy_dot_denominator)
        k11 = _cauchy_dot(a1 * b1, denominator=cauchy_dot_denominator)
        return c * (k00 - k01 * (1.0 / (1.0 + k11)) * k10)

    @property
    def K(self) -> torch.Tensor:  # noqa
        """K convolutional filter."""
        at_roots = self._compute_roots()
        out = ifft(at_roots, n=self.l_max, dim=-1)
        conv = torch.stack([i[self.ifft_order] for i in out]).real
        return conv.unsqueeze(0)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            u (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_INPUT]``

        Returns:
            y (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_OUTPUT]``

        """
        return _non_circular_convolution(u, K=self.K) + (self.D * u)




# S4 Block:

def _make_norm(d_model: int, norm_type: Optional[str]) -> nn.Module:
    if norm_type is None:
        return nn.Identity()
    elif norm_type == "layer":
        return nn.LayerNorm(d_model)
    elif norm_type == "batch":
        return TemporalAdapter(nn.BatchNorm1d(d_model))
    else:
        raise ValueError(f"Unsupported norm type '{norm_type}'")

class S4Block(nn.Module):
    """S4 Block.

    Applies ``S4Layer()``, followed by an activation
    function, dropout, linear layer, skip connection and
    layer normalization.

    Args:
        d_model (int): number of internal features
        n (int): dimensionality of the state representation
        l_max (int): length of input signal
        p_dropout (float): probability of elements being set to zero
        activation (Type[nn.Module]): activation function to use after
            ``S4Layer()``.
        norm_type (str, optional): type of normalization to use.
            Options: ``batch``, ``layer``, ``None``.
        norm_strategy (str): position of normalization relative to ``S4Layer()``.
            Must be "pre" (before ``S4Layer()``), "post" (after ``S4Layer()``)
            or "both" (before and after ``S4Layer()``).
        pooling (nn.AvgPool1d, nn.MaxPool1d, optional): pooling method to use
            following each ``S4Block()``.

    """

    def __init__(
        self,
        d_model: int,
        n: int,
        l_max: int,
        p_dropout: float = 0.0,
        activation: Type[nn.Module] = nn.GELU,
        norm_type: Optional[str] = "layer",
        norm_strategy: str = "post",
        pooling: Optional[nn.AvgPool1d | nn.MaxPool1d] = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n = n
        self.l_max = l_max
        self.p_dropout = p_dropout
        self.activation = activation
        self.norm_type = norm_type
        self.norm_strategy = norm_strategy
        self.pooling = pooling

        if norm_strategy not in ("pre", "post", "both"):
            raise ValueError(f"Unexpected norm_strategy, got '{norm_strategy}'")

        self.pipeline = SequentialWithResidual(
            (
                _make_norm(d_model, norm_type=norm_type)
                if norm_strategy in ("pre", "both")
                else nn.Identity()
            ),
            S4Layer(d_model, n=n, l_max=l_max),
            activation(),
            nn.Dropout(p_dropout),
            nn.Linear(d_model, d_model, bias=True),
            Residual(),
            (
                _make_norm(d_model, norm_type=norm_type)
                if norm_strategy in ("post", "both")
                else nn.Identity()
            ),
            TemporalAdapter(pooling) if pooling else nn.Identity(),
            nn.Dropout(p_dropout),
        )

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            u (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_INPUT]``

        Returns:
            y (torch.Tensor): a tensor of the form ``[BATCH, SEQ_LEN, D_OUTPUT]``

        """
        return self.pipeline(u)
