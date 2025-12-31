import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os

DATA_PATH = r"c:\Users\YAMINI\Downloads\archive\CMaps"
TRAIN_FILE = "train_FD001.txt"
TEST_FILE = "test_FD001.txt"
RUL_FILE = "RUL_FD001.txt"

COLUMNS = ['unit', 'time', 'os1', 'os2', 'os3'] + [f's{i}' for i in range(1, 22)]
SENSOR_COLS = [f's{i}' for i in range(1, 22)]
MAX_RUL = 125
WINDOW_SIZE = 30

def load_data(file_path):
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=COLUMNS)
    return df

def process_rul(df, max_rul=MAX_RUL):
    max_cycles = df.groupby('unit')['time'].max().reset_index()
    max_cycles.columns = ['unit', 'max_time']
    df = df.merge(max_cycles, on='unit', how='left')
    df['RUL_raw'] = df['max_time'] - df['time']
    df['RUL'] = df['RUL_raw'].clip(upper=max_rul)
    df.drop(['max_time', 'RUL_raw'], axis=1, inplace=True)
    return df

def scale_data(train_df, test_df, sensor_cols):
    scaler = MinMaxScaler()
    train_df[sensor_cols] = scaler.fit_transform(train_df[sensor_cols])
    test_df[sensor_cols] = scaler.transform(test_df[sensor_cols])
    return train_df, test_df, scaler

def create_windows(df, window_size, sensor_cols):
    dataset_X = []
    dataset_y = []
    unique_units = df['unit'].unique()
    
    for unit in unique_units:
        unit_data = df[df['unit'] == unit]
        if len(unit_data) < window_size:
            continue
        data_values = unit_data[sensor_cols].values
        rul_values = unit_data['RUL'].values
        
        for i in range(window_size, len(unit_data)):
            dataset_X.append(data_values[i-window_size:i, :])
            dataset_y.append(rul_values[i])
            
    return np.array(dataset_X), np.array(dataset_y)

def prepare_data():
    print("Loading Data...")
    train_path = os.path.join(DATA_PATH, TRAIN_FILE)
    test_path = os.path.join(DATA_PATH, TEST_FILE)
    
    train_df = load_data(train_path)
    test_df = load_data(test_path)
    
    print("Processing RUL...")
    train_df = process_rul(train_df)
    
    print("Scaling Data...")
    train_df, test_df, scaler = scale_data(train_df, test_df, SENSOR_COLS)
    
    print("Windowing Training Data...")
    X_train, y_train = create_windows(train_df, WINDOW_SIZE, SENSOR_COLS)
    
    print("Preparing Test Data...")
    X_test_list = []
    valid_test_idxs = [] 
    unique_test_units = test_df['unit'].unique()
    
    for i, unit in enumerate(unique_test_units):
        unit_data = test_df[test_df['unit'] == unit]
        if len(unit_data) >= WINDOW_SIZE:
            data_values = unit_data[SENSOR_COLS].values
            last_window = data_values[-WINDOW_SIZE:, :]
            X_test_list.append(last_window)
            valid_test_idxs.append(i)
            
    X_test = np.array(X_test_list)
    
    y_true_rul = pd.read_csv(os.path.join(DATA_PATH, RUL_FILE), sep=r'\s+', header=None, names=['RUL'])
    y_test_true = y_true_rul['RUL'].values
    y_test = y_test_true[valid_test_idxs]
    
    return X_train, y_train, X_test, y_test, WINDOW_SIZE, len(SENSOR_COLS)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test, _, _ = prepare_data()
    print(f"Data Ready. Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
