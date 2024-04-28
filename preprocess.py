import math
import numpy as np
from sklearn.model_selection import train_test_split
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
    "levels": [500, 700, 850],
    "time_slice": slice('2020-01-01', '2020-12-31'),
    "lat_slice": slice(30,50),
    "long_slice": slice(70,90),
}



def preprocess_data(split_percentage: float, batch_size=10, use_level=False,  window_size=10):
    obs_path = 'gs://weatherbench2/datasets/era5/1959-2022-6h-64x32_equiangular_conservative.zarr'
    data = xr.open_zarr(obs_path)

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
            labels.append(sequence[-1])
    inputs = np.stack(inputs, axis=0)
    labels = np.stack(labels, axis=0)


    # (num_batches, batch_size, sequence_len, level, lon, lat, features)
    # (num_batches, sequence_len, batch_size, features, lat, lon)

    #split into training and testing
    num_samples = len(inputs)
    training_size = math.floor(split_percentage*num_samples)
    X_train = inputs[0:training_size]
    X_test = inputs[training_size::]
    Y_train = labels[0:training_size]
    Y_test = labels[training_size::]


    print("x train:", X_train.shape)
    print("x test:", X_test.shape)
    print("y train:", Y_train.shape)
    print("y test:", Y_test.shape)
    
    return X_train, X_test, Y_train, Y_test

    # np.save('../data/test_data_array.npy', dataset)


preprocess_data(0.8)
