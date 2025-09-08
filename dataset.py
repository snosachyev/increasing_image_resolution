import albumentations as A
import numpy as np
import keras

from keras import utils

from utils import noise_image


# Генератор для перебора данных (в виде массивов Numpy)
class datasetGenerator(keras.utils.Sequence):

    def __init__(self, batch_size, input_img_path, img_size=(32, 32), validation=False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_path = input_img_path
        self.validation = validation

    def __len__(self):
        """Возвращает число мини-батчей обучающей выборки"""
        return len(self.input_img_path) // self.batch_size

    def __getitem__(self, idx):
        """Возвращает input соответствующий индексу пакета idx"""

        # Формируем пакеты из ссылок путем среза длинной в batch_size и возвращаем пакет по индексу
        batch_input_img_path = self.input_img_path[idx * self.batch_size:(idx + 1) * self.batch_size]
        # Создадим массив numpy, заполненный нулями, для входных данных формы (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 3) и типа данных float32
        x = np.zeros((self.batch_size, *self.img_size, 3), dtype="float32")

        # Создадим массив numpy, заполненный нулями, для выходных данных формы (BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH,
        # 3) и типа данных float32
        y = np.zeros((self.batch_size, *self.img_size, 3), dtype="float32")

        # В цикле заполняем массивы с изображениями x
        # Перебираем пакеты из путей batch_input_img_path к изображениям
        # возвращает для нескольких последовательностей список кортежей из элементов последовательностей с одинаковыми индексами
        for _, path in enumerate(batch_input_img_path):

            # Загружаем изображение и маску используя путь файловой системы
            img = np.array(utils.load_img(path, target_size=self.img_size, color_mode='rgb'))  # 3 канала для
            # изображения
            if self.validation:
                # Применяем аугментацию для проверочной выборки (p - вероятность применения, 0.5 - для каждого второго изображения)
                transform = A.Compose([  # определяем функцию аугментации
                    #A.Flip(p=0.5),  # Отражение изображения по горизонтали и вертикали
                    A.RandomRotate90(p=0.5)  # Случайный поворот на 90 градусов
                ])
                transformed = transform(image=img)  # применяем функцию аугментации в изображению и маске
                img = transformed["image"]

            img_norm = img / 255.0

            y[_] = img_norm  # нормализуем изображение

            x[_] = noise_image(img_norm)
            y[_] = img_norm

        return x, y
