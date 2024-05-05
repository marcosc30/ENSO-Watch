import math
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
import xarray as xr

selection = {
    'variables': [
        'geopotential',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind',
        'vertical_velocity',
        '10m_wind_speed',
        'total_precipitation_6hr',
        'total_cloud_cover',
        '2m_temperature',
        'specific_humidity',
        'surface_pressure',
        'toa_incident_solar_radiation',
        'total_column_water_vapour'
    ],
    "levels": [500],
    "time_slice": slice('2016-01-01', '2017-01-01'),
    "lat_slice": slice(30,50),
    "long_slice": slice(70,90),
}


def preprocess_data(split_percentage=0.8, batch_size=20, use_level=False,  window_size=10):
    obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr'
    data = xr.open_zarr(obs_path)

    print("preprocessing")
    print("dataset shape: ", data.sizes)

    data = data[selection['variables']].sel(level=selection['levels'], time=selection['time_slice'],
                                            latitude=selection['lat_slice'], longitude=selection['long_slice'])
    # data = data[selection['variables']].sel(level=selection['levels'],
    #                                     latitude=selection['lat_slice'], longitude=selection['long_slice'])

    time_size = data.sizes['time']
    level_size = data.sizes['level']
    lon_size = data.sizes['longitude']
    lat_size = data.sizes['latitude']
    
    feature_size = len(selection['variables'])

    
    dataset_shape = (time_size, level_size, lon_size, lat_size) if use_level else (time_size, lon_size, lat_size)
    time_arrays = []
    for i, var_name in enumerate(data.data_vars):
        var_data = data[var_name].values

        num_dims = len(var_data.shape)
        time_array = np.empty(dataset_shape, dtype=var_data.dtype)
        has_level_size = num_dims == 4
        if use_level:
            if has_level_size:
                time_array[:] = var_data
            else:
                time_array[:] = np.expand_dims(var_data, axis=1)
        else:
            if has_level_size:
                time_array[:] = var_data[:, 0, :, :]
            else:
                time_array[:] = var_data

        
        time_arrays.append(time_array)

    dataset = np.stack(time_arrays, axis=-1)
    print("processed dataset shape:", dataset.shape)

    #processing to time series
    default_intervals = [-120, -56, -28, -12, -8, -4, -3, -2, -1, 0, 4]
    #todo: handle longer window size

    inputs = []
    labels = []
    for i in range(len(dataset)):
        sequence = []
        for interval in default_intervals:
            index = i+interval
            if 0 <= index < len(dataset):
                sequence.append(dataset[index])
        
        if len(sequence) == len(default_intervals):
            inputs.append(sequence[0:len(sequence)-1])
            labels.append([sequence[-1]])
    inputs = np.stack(inputs, axis=0)
    labels = np.stack(labels, axis=0)


    num_samples = len(inputs)
    # num_batches = num_samples // batch_size
    # sequence_len = len(default_intervals) - 1
    
    # inputs_truncated = inputs[:total_samples - total_samples % batch_size]
    # labels_truncated = labels[:total_samples - total_samples % batch_size]
    
    if use_level:
        # # (total_samples, sequence_len, level, lon, lat, features)
        # new_inputs_shape = (num_batches, batch_size, sequence_len, level_size, lon_size, lat_size, feature_size)
        # new_labels_shape = (num_batches, batch_size, 1, level_size, lon_size, lat_size, feature_size)
        
        # # (num_batches, batch_size, sequence_len, level, lon, lat, features)
        # inputs = inputs_truncated.reshape(new_inputs_shape)
        # labels = labels_truncated.reshape(new_labels_shape)

        # # (num_batches, sequence_len, batch_size, features, lat, lon, level)
        # inputs = np.transpose(inputs, (0, 2, 1, 6, 5, 4, 3))
        # labels = np.transpose(labels, (0, 2, 1, 6, 5, 4, 3))

        # (total_samples, sequence_len, features, lat, lon, level)
        inputs = np.transpose(inputs, (0, 1, 5, 4, 3, 2))
        labels = np.transpose(labels, (0, 1, 5, 4, 3, 2))


    else:
        # # (total_samples, sequence_len, lon, lat, features)
        # new_inputs_shape = (num_batches, batch_size, sequence_len, lon_size, lat_size, feature_size)
        # new_labels_shape = (num_batches, batch_size, 1, lon_size, lat_size, feature_size)
        
        # # (num_batches, batch_size, sequence_len, lon, lat, features)
        # inputs = inputs_truncated.reshape(new_inputs_shape)
        # labels = labels_truncated.reshape(new_labels_shape)
    
        # # (num_batches, sequence_len, batch_size, features, lat, lon)
        # inputs = np.transpose(inputs, (0, 2, 1, 5, 4, 3))
        # labels = np.transpose(labels, (0, 2, 1, 5, 4, 3))

        # (total_samples, sequence_len, features, lat, lon)
        inputs = np.transpose(inputs, (0, 1, 4, 3, 2))
        labels = np.transpose(labels, (0, 1, 4, 3, 2))

    # print('BEFORE')
    # print(inputs[0])
    # print(labels[0])
    # normalization
    flattened_inputs = inputs.reshape(-1, inputs.shape[2])
    flattened_labels = labels.reshape(-1, labels.shape[2])

    inputs_mean = np.mean(flattened_inputs, axis=0).reshape(1,1,12,1,1)
    labels_mean = np.mean(flattened_labels, axis=0).reshape(1,1,12,1,1)
    inputs_std = np.std(flattened_inputs, axis=0).reshape(1,1,12,1,1)
    labels_std = np.std(flattened_labels, axis=0).reshape(1,1,12,1,1)

    normalized_inputs = (inputs - inputs_mean) / inputs_std 
    normalized_labels = (labels - labels_mean) / labels_std

    inputs = normalized_inputs.reshape(inputs.shape)
    labels = normalized_labels.reshape(labels.shape)

    # normalization
    # input_mean = np.mean(inputs)
    # input_std = np.std(inputs)
    # label_mean = np.mean(labels)
    # label_std = np.std(labels)

    # normalized_inputs = (inputs - input_mean) / input_std
    # normalized_labels = (labels - label_mean) / label_std

    # print("NORMALIZED")
    # print(inputs[0])
    # print(labels[0])

    #split into training and testing
    training_size = math.floor(split_percentage*num_samples)
    X_train = torch.tensor(inputs[0:training_size], dtype=torch.float32)
    X_test = torch.tensor(inputs[training_size::], dtype=torch.float32)
    Y_train = torch.tensor(labels[0:training_size], dtype=torch.float32)
    Y_test = torch.tensor(labels[training_size::], dtype=torch.float32)

    # print("x train:", X_train.shape)
    # print("x test:", X_test.shape)
    # print("y train:", Y_train.shape)
    # print("y test:", Y_test.shape)

    # X: (num_samples, sequence_len, features, lat, lon)
    # Y: (num_samples, 1, features, lat, lon)
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)

    torch.save(train_dataset, './data/train_dataset_norm_one_year.pth')
    torch.save(test_dataset, './data/test_dataset_norm_one_year.pth')

    return train_dataset, test_dataset
    

    # np.save('../data/test_data_array.npy', dataset)


preprocess_data(0.8)