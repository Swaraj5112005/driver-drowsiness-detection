from tensorflow import keras
from tensorflow.keras import layers
import config

S = config.IMG_SIZE


def _front():
    inp = keras.Input((S, S, 1))
    x = layers.Rescaling(1 / 255.0)(inp)
    x = keras.Sequential([
        layers.RandomRotation(0.08), layers.RandomZoom(0.1),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomBrightness(0.25, value_range=(0, 1)),
        layers.RandomContrast(0.25)], name="augment")(x)   # only active while training
    return inp, x


def simple_cnn():
    """Baseline: 2 conv blocks + dense head."""
    inp, x = _front()
    for f in (16, 32):
        x = layers.Conv2D(f, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(x), name="simple_cnn")


def improved_cnn():
    """BatchNorm + 3 conv blocks + global pooling + dropout."""
    inp, x = _front()
    for f in (32, 64, 128):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(32, activation="relu")(x)
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(x), name="improved_cnn")


def mobilenet():
    """Optional transfer-learning comparison (MobileNetV2, ImageNet weights)."""
    inp, x = _front()
    x = layers.Resizing(96, 96)(x)
    x = layers.Concatenate()([x, x, x])            # gray -> 3 channels
    x = layers.Rescaling(2.0, offset=-1.0)(x)      # [0,1] -> [-1,1]
    base = keras.applications.MobileNetV2((96, 96, 3), include_top=False, weights="imagenet")
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    return keras.Model(inp, layers.Dense(1, activation="sigmoid")(x), name="mobilenetv2")


def build(arch):
    return {"simple": simple_cnn, "improved": improved_cnn, "mobilenet": mobilenet}[arch]()
