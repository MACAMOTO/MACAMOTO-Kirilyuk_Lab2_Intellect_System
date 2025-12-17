
# Задача 2: Классификация IMDb отзывов

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense

# 1. Загрузка датасета IMDb (реальный текст)
ds_train, ds_test = tfds.load('imdb_reviews', split=['train', 'test'], as_supervised=True)

# Преобразуем в списки Python
train_texts, train_labels = [], []
for text, label in tfds.as_numpy(ds_train):
    train_texts.append(text.decode('utf-8'))
    train_labels.append(label)

test_texts, test_labels = [], []
for text, label in tfds.as_numpy(ds_test):
    test_texts.append(text.decode('utf-8'))
    test_labels.append(label)

# 2. Токенизация и паддинг
vocab_size = 10000
max_len = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_texts)

X_train = tokenizer.texts_to_sequences(train_texts)
X_test = tokenizer.texts_to_sequences(test_texts)

X_train = pad_sequences(X_train, maxlen=max_len, padding='post')
X_test = pad_sequences(X_test, maxlen=max_len, padding='post')

y_train = tf.convert_to_tensor(train_labels)
y_test = tf.convert_to_tensor(test_labels)

# 3. Модель LSTM
model_lstm = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n--- Обучение модели LSTM ---")
history_lstm = model_lstm.fit(
    X_train, y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

lstm_acc = model_lstm.evaluate(X_test, y_test, verbose=0)[1]
print(f"Точность LSTM на тесте: {lstm_acc:.4f}")

# 4. Модель GRU
model_gru = Sequential([
    Embedding(input_dim=vocab_size, output_dim=32, input_length=max_len),
    GRU(32),
    Dense(1, activation='sigmoid')
])

model_gru.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("\n--- Обучение модели GRU ---")
history_gru = model_gru.fit(
    X_train, y_train,
    epochs=3,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

gru_acc = model_gru.evaluate(X_test, y_test, verbose=0)[1]
print(f"Точность GRU на тесте: {gru_acc:.4f}")

# 5. Сравнение моделей
print("\nСравнение точности на тестовой выборке:")
print(f"LSTM Accuracy: {lstm_acc:.4f}")
print(f"GRU Accuracy:  {gru_acc:.4f}")
