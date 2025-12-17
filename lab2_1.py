# Прогнозирование временных рядов (Unit Price)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense

# 1. Загрузка данных
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(url)

series = df['Temp'].values.reshape(-1, 1)

plt.figure(figsize=(10, 4))
plt.plot(series)
plt.title("Временной ряд: Temp")
plt.show()

# 2. Нормализация и создание оконных выборок
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series)

window_size = 20 
def create_dataset(data, window_size):
    X, y = [], []
    for i in range(window_size, len(data)):
        X.append(data[i-window_size:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(series_scaled, window_size)


split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# 3. Построение и обучение моделей
def build_model(model_type, window_size):
    model = Sequential()
    if model_type == 'SimpleRNN':
        model.add(SimpleRNN(50, activation='tanh', input_shape=(window_size,1)))
    elif model_type == 'LSTM':
        model.add(LSTM(50, activation='tanh', input_shape=(window_size,1)))
    elif model_type == 'GRU':
        model.add(GRU(50, activation='tanh', input_shape=(window_size,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

models = {}
histories = {}
for model_type in ['SimpleRNN', 'LSTM', 'GRU']:
    print(f"\n--- Обучение {model_type} ---")
    model = build_model(model_type, window_size)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32,
                        validation_data=(X_test, y_test), verbose=0)
    models[model_type] = model
    histories[model_type] = history

# 4. Визуализация потерь
plt.figure(figsize=(12, 6))
for model_type in ['SimpleRNN', 'LSTM', 'GRU']:
    plt.plot(histories[model_type].history['val_loss'], label=f'{model_type} Val Loss')
plt.title('Сравнение потерь на валидации (MSE)')
plt.xlabel('Эпоха')
plt.ylabel('MSE')
plt.legend()
plt.show()

# 5. Прогноз и MSE
plt.figure(figsize=(15, 8))
for i, model_type in enumerate(['SimpleRNN', 'LSTM', 'GRU']):
    y_pred = models[model_type].predict(X_test)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
    y_pred_inv = scaler.inverse_transform(y_pred)
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    plt.subplot(2, 2, i+1)
    plt.plot(y_test_inv, label='Истинные значения')
    plt.plot(y_pred_inv, label='Прогноз')
    plt.title(f'{model_type} (MSE: {mse:.4f})')
    plt.legend()
plt.tight_layout()
plt.show()
