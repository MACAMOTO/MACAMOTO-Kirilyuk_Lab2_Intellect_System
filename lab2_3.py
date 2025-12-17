# =============================
# Прогнозирование временных рядов без TensorFlow
# =============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# -----------------------------
# 1. Загрузка данных из интернета
# -----------------------------
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"
df = pd.read_csv(url)

# Используем колонку "Temp" как временной ряд
series = df['Temp'].values.reshape(-1, 1)

plt.figure(figsize=(10, 4))
plt.plot(series)
plt.title("Временной ряд: Temp")
plt.show()

# -----------------------------
# 2. Нормализация и создание оконных выборок
# -----------------------------
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

# Разделение на train и test
split = int(len(X) * 0.8)
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# -----------------------------
# 3. Построение и обучение моделей (имитация RNN через линейную регрессию)
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# Прогноз
y_pred = model.predict(X_test)

# Обратное масштабирование
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1,1))

# MSE
mse = mean_squared_error(y_test_inv, y_pred_inv)
print(f"MSE на тесте: {mse:.4f}")

# -----------------------------
# 4. Визуализация прогнозов
# -----------------------------
plt.figure(figsize=(12, 5))
plt.plot(y_test_inv, label='Истинные значения')
plt.plot(y_pred_inv, label='Прогноз')
plt.title(f'Прогноз временного ряда (MSE={mse:.4f})')
plt.xlabel('Время')
plt.ylabel('Температура')
plt.legend()
plt.show()
