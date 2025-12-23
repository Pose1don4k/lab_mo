import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Уменьшает вывод логов TensorFlow

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, datasets

# 1. Загрузка и предобработка данных MNIST
# Набор данных содержит 60k обучающих и 10k тестовых изображений цифр 28x28[citation:1]
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print(f"Загружены данные: x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

# Масштабирование значений пикселей к диапазону [0, 1] и добавление измерения для канала цвета (grayscale)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = tf.expand_dims(x_train, -1)  # Формат (60000, 28, 28, 1)
x_test = tf.expand_dims(x_test, -1)    # Формат (10000, 28, 28, 1)

print(f"Новые формы: x_train: {x_train.shape}, x_test: {x_test.shape}")

# 2. Определение архитектуры сверточной нейронной сети (CNN)
model = keras.Sequential([
    # Первый сверточный блок
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Второй сверточный блок для извлечения более сложных признаков
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # Преобразование 2D-признаков в вектор для полносвязных слоев
    layers.Flatten(),
    
    # Полносвязные слои для классификации
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Регуляризация для борьбы с переобучением
    layers.Dense(10, activation='softmax')  # Выходной слой для 10 классов (цифры 0-9)
])

# 3. Компиляция модели
# SparseCategoricalCrossentropy подходит для целочисленных меток[citation:7]
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']  # Следим за точностью классификации[citation:3][citation:7]
)

# Вывод сводки архитектуры
model.summary()

# 4. Обучение модели
print("\nНачало обучения модели...")
history = model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=5,  
    validation_split=0.1
)


model.save('cnn_mnist_model.keras')
print(f"Модель успешно сохранена в файл: cnn_mnist_model.keras")


print("\nОценка точности модели на тестовых данных...")
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print(f"Итоговая точность на тестовых данных: {test_accuracy:.4f}")
