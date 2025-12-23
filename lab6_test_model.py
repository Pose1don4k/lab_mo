import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np

def load_and_test_model():
    """Загружает модель из файла и предоставляет интерфейс для проверки."""
    
    # 1. Загрузка модели из файла
    # Функция load_model восстанавливает модель в точном состоянии[citation:5][citation:9]
    model_path = 'cnn_mnist_model.keras'
    
    if not os.path.exists(model_path):
        print(f"Ошибка: Файл модели '{model_path}' не найден.")
        print("Сначала выполните программу lab6_train_model.py для обучения и сохранения модели.")
        return
    
    print(f"Загрузка модели из файла: {model_path}")
    loaded_model = keras.models.load_model(model_path)
    print("Модель успешно загружена.")
    
    # 2. Загрузка тестовых данных для проверки
    # Используем встроенный датасет для удобства[citation:1]
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Такая же предобработка, как и при обучении
    x_test = x_test.astype("float32") / 255.0
    x_test = tf.expand_dims(x_test, -1)
    
    # 3. Основной цикл проверки
    while True:
        print("\n" + "="*50)
        print("МЕНЮ ПРОВЕРКИ ТОЧНОСТИ КЛАССИФИКАТОРА")
        print("="*50)
        print("1. Оценить точность на всем тестовом наборе (10 000 изображений)")
        print("2. Протестировать на одном случайном изображении")
        print("3. Выйти из программы")
        
        choice = input("\nВыберите действие (1-3): ").strip()
        
        if choice == '1':
            # Оценка точности на всем тестовом наборе данных[citation:7]
            print("Выполняется оценка на тестовом наборе...")
            test_loss, test_accuracy = loaded_model.evaluate(x_test, y_test, verbose=0)
            print(f"Результаты оценки:")
            print(f"  Потери (loss): {test_loss:.4f}")
            print(f"  Точность (accuracy): {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            
        elif choice == '2':
            # Тестирование на одном случайном изображении
            idx = np.random.randint(0, len(x_test))
            test_image = x_test[idx]
            true_label = y_test[idx]
            
            # Добавляем размерность батча для predict
            prediction = loaded_model.predict(tf.expand_dims(test_image, 0), verbose=0)
            predicted_label = np.argmax(prediction[0])
            confidence = np.max(prediction[0]) * 100
            
            print(f"\nТестирование случайного изображения #{idx}:")
            print(f"  Истинная цифра: {true_label}")
            print(f"  Предсказанная цифра: {predicted_label}")
            print(f"  Уверенность модели: {confidence:.2f}%")
            
            if predicted_label == true_label:
                print("  ✓ Предсказание верное!")
            else:
                print("  ✗ Предсказание неверное.")
                
            # Простая ASCII-визуализация изображения
            if input("Показать изображение? (y/n): ").lower() == 'y':
                # Используем исходные данные без нормализации
                img_array = (test_image.numpy().squeeze() * 255).astype(int)
                for i in range(28):
                    row = ''.join(['  ' if pixel < 128 else '██' for pixel in img_array[i]])
                    print(row)
                    
        elif choice == '3':
            print("Выход из программы.")
            break
            
        else:
            print("Неверный выбор. Пожалуйста, введите 1, 2 или 3.")

if __name__ == "__main__":
    load_and_test_model()
