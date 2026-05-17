import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Загрузка данных из файлов
# dataIn.txt: N строк, по 12 входных признаков (int или float)
# dataOut.txt: N строк, по 2 выхода (one-hot или 0/1-метки)
X = np.loadtxt("dataIn.txt")
Y = np.loadtxt("dataOut.txt")

# Если выходы даны как 0/1 классы (одна колонка),
# можно привести их к one-hot (раскомментировать при необходимости):
# if Y.ndim == 1 or Y.shape[1] == 1:
#     num_classes = int(np.max(Y)) + 1
#     Y = keras.utils.to_categorical(Y.astype(int), num_classes=num_classes)

# 2. Разбиение на обучающую и тестовую выборки
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42, shuffle=True
)

# 3. Масштабирование входных признаков
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Построение модели MLP
# Пример: вход 12, два скрытых слоя, выход 2 (для двух классов с softmax)
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(8, activation='relu'),
    # Если Y — one-hot из 2 классов:
    keras.layers.Dense(Y_train.shape[1], activation='softmax')
])

# 5. Компиляция модели
# Для многоклассовой классификации с one-hot:
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Если у вас бинарная классификация с выходом (N, 1) и sigmoid,
# используйте: loss='binary_crossentropy' и последнюю Dense(1, activation='sigmoid')

# 6. Обучение модели
history = model.fit(
    X_train_scaled,
    Y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test_scaled, Y_test),
    verbose=1
)

# 7. Предсказания и оценка точности
# Для one-hot выходов переводим индекс максимального элемента в метку класса
Y_pred_proba = model.predict(X_test_scaled)
Y_pred = np.argmax(Y_pred_proba, axis=1)
Y_true = np.argmax(Y_test, axis=1)

acc = accuracy_score(Y_true, Y_pred)
print("Test accuracy:", acc)

# 8. Графики ошибок и точности
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss over epochs')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training accuracy')
plt.plot(history.history['val_accuracy'], label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy over epochs')

plt.tight_layout()
plt.show()