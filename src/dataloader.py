import os

import numpy as np
import tensorflow as tf

IMG_SIZE = (224, 224)


def multihot_labels(genre_lists, genre_names):
    """Konversi list genre per baris jadi matrix multi-hot [n_samples, n_genres]."""
    return np.array([[1.0 if g in gl else 0.0 for g in genre_names]
                     for gl in genre_lists], dtype="float32")


def build_dataset(posters_dir, filenames, labels, batch_size=32, training=False, seed=42):
    """
    tf.data pipeline: read file -> decode -> resize 224x224 -> (augment) -> batch.

    Augmentasi sengaja terbatas (flip horizontal + brightness ringan) karena
    teks dan komposisi poster adalah sinyal penting untuk genre.
    """
    paths = [os.path.join(posters_dir, f) for f in filenames]
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        return img, label

    def _augment(img, label):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.1)
        return img, label

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(len(paths), seed=seed)
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
