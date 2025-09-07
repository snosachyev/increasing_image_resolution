import os

import numpy as np
# Загрузка необходимых библиотек
import matplotlib.pyplot as plt

from keras import utils as keras_utils


def update_history(history, file_name):
    try:
        cnn_history = np.load(f'{file_name}.npy', allow_pickle='TRUE').item()
    except FileNotFoundError:
        cnn_history = history
    else:
        cnn_history['loss'].extend(history['loss'])
        cnn_history['val_loss'].extend(history['val_loss'])
        cnn_history['learning_rate'].extend(history['learning_rate'])
    np.save(f'{file_name}.npy', cnn_history)

    return cnn_history


def show_loss(history):
    plt.figure(figsize=(10, 6))  # Optional: Adjust figure size
    plt.plot(history['loss'])
    plt.title('CNN autoencoder Training Loss')
    plt.xlabel('Epochs (or Iterations)')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()


# Метрики качества (MSE и PSNR)
def psnr(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    if mse == 0:
        return 100
    max_pixel = 1.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def create_input_img_path(folder_path):
    input_img_path = []

    for dir_name in os.listdir(folder_path):
        image_dir = os.path.join(folder_path, dir_name)
        if os.path.isdir(image_dir):
            for i, filename in enumerate(os.listdir(image_dir)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                    image_path = os.path.join(image_dir, filename)
                    input_img_path.append(image_path)

    return input_img_path


def load_data(images_path, target_size=(32, 32)):
    dataset = []
    for image_path in images_path:
        img = keras_utils.load_img(image_path, target_size=target_size)  # Optional: resize the image
        img_array = keras_utils.img_to_array(img)
        dataset.append(img_array)
    return np.asarray(dataset, dtype=np.float32)
