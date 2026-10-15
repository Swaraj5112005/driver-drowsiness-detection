import tensorflow as tf
import config

S = config.IMG_SIZE


def make_ds(df, training, batch=64):
    ds = tf.data.Dataset.from_tensor_slices((df.path.values, df.label.values.astype("float32")))
    if training:
        ds = ds.shuffle(min(len(df), 20000), seed=config.SEED)

    def load(p, y):
        img = tf.io.decode_image(tf.io.read_file(p), channels=1, expand_animations=False)
        img.set_shape([None, None, 1])
        return tf.image.resize(img, (S, S)), y

    return ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)
