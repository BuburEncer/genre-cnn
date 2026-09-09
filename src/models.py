import tensorflow as tf
from tensorflow.keras import layers


def _compile(model):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="acc"),
            tf.keras.metrics.AUC(multi_label=True, name="auc"),
        ],
    )


def build_baseline_cnn(num_classes=7, input_shape=(224, 224, 3)):
    """CNN kecil from-scratch sebagai baseline pembanding di bab hasil."""
    model = tf.keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Rescaling(1.0 / 255),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="sigmoid"),
    ])
    _compile(model)
    return model


def build_transfer_model(num_classes=7, input_shape=(224, 224, 3), trainable=False):
    """
    MobileNetV2 pre-trained ImageNet sebagai feature extractor + head sigmoid.

    preprocess_input (scaling ke [-1, 1]) disertakan di dalam model supaya
    pipeline data cukup memberi gambar uint8/float mentah 0-255.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = trainable

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    _compile(model)
    return model
