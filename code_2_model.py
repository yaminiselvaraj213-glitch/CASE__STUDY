from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D

FILTERS = 64
KERNEL_SIZE = 3
LSTM_UNITS = 64

def build_model(input_shape):
    model = Sequential()
    
    model.add(Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, activation='relu', input_shape=input_shape, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.1))
    
    model.add(LSTM(LSTM_UNITS, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(LSTM_UNITS // 2, return_sequences=False))
    model.add(Dropout(0.2))
    
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

if __name__ == "__main__":
    model = build_model((30, 21))
    model.summary()
