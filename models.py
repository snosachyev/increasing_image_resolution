from tensorflow.keras import layers, models

# =========================
# МОДЕЛЬ 1: CNN-АВТОКОДИРОВЩИК
# =========================

def create_cnn_autoencoder_model(input_shape=(32, 32, 3), layer_size=64, num_layers=2):
    input_img = layers.Input(shape=input_shape)
    # Кодировщик
    x = input_img
    for layer in range(1, num_layers + 1):
        x = layers.Conv2D(layer_size * layer, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Де кодировщик
    for layer in range(num_layers, 0, -1):
        x = layers.Conv2D(layer_size * layer, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)

    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    return models.Model(input_img, decoded)


# =========================
# МОДЕЛЬ 2: RESIDUAL AUTOENCODER
# =========================

def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(filters, (3, 3), padding="same")(x)
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    return x


def create_residual_block_model(input_shape=(32, 32, 3), layer_size=64, num_layers=2):
    input_res = layers.Input(shape=input_shape)

    r = input_res

    # Кодировщик
    for layer in range(1, num_layers + 1):
        r = layers.Conv2D(layer_size * layer, (3, 3), activation='relu', padding='same')(r)
        r = layers.MaxPooling2D((2, 2), padding='same')(r)
        r = residual_block(r, layer_size * layer)

    # Декодировщик
    for layer in range(num_layers, 0, -1):
        r = layers.Conv2D(layer_size * layer, (3, 3), activation='relu', padding='same')(r)
        r = layers.UpSampling2D((2, 2))(r)
        r = residual_block(r, layer_size * layer)

    residual_decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(r)

    return models.Model(input_res, residual_decoded)
