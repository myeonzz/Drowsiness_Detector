"""Train the two binary models (eye: Open/Closed, mouth: no_yawn/yawn), save
them as .keras and convert to float + INT8 TensorFlow Lite.

    python train.py            # both models
    python train.py eye        # only one
"""
import json
import sys

import cv2
import numpy as np

import common
import config

import keras
import tensorflow as tf
from keras import layers


# ------------------------------------------------------------------ data ---
def build_cache(name):
    """Read images, crop (mouth model: mouth only), resize to 96x96 gray,
    cache as .npz. The mouth cache also keeps each photo's MAR."""
    out = config.CACHE_DIR / f"{name}.npz"
    if out.exists():
        return
    config.CACHE_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    finder = common.FaceFinder(video=False) if name == "mouth" else None
    X, y, mar, dropped = [], [], [], {}
    for label, folder in enumerate(config.MODELS[name]):
        files = sorted((config.DATA_DIR / folder).glob("*"))
        dropped[folder] = {"total": len(files), "dropped": 0}
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if img is None:
                dropped[folder]["dropped"] += 1
                continue
            if finder is not None:                     # same crop as the demo
                lm = finder.find(img)
                box = common.mouth_box(lm, img.shape) if lm is not None else None
                if box is None:
                    dropped[folder]["dropped"] += 1
                    continue
                img = common.crop(img, box)
                mar.append(common.mouth_aspect_ratio(lm))
            X.append(common.preprocess(img))
            y.append(label)
    if finder is not None:
        finder.close()
    extra = {"mar": np.array(mar, np.float32)} if mar else {}
    np.savez_compressed(out, X=np.array(X, np.uint8), y=np.array(y, np.int64), **extra)
    for folder, d in dropped.items():
        why = "no face found" if name == "mouth" else "unreadable"
        print(f"[{name}] {folder}: {d['total'] - d['dropped']} kept, "
              f"{d['dropped']} dropped ({why}) of {d['total']}")
    (config.RESULTS_DIR / f"{name}_dropped.json").write_text(json.dumps(dropped, indent=2))


class AugmentedData(keras.utils.PyDataset):
    """Yields freshly augmented batches every epoch."""

    def __init__(self, X, y, seed, **kwargs):
        super().__init__(**kwargs)
        self.X, self.y = X, y
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / config.BATCH_SIZE))

    def __getitem__(self, i):
        idx = self.order[i * config.BATCH_SIZE:(i + 1) * config.BATCH_SIZE]
        batch = np.stack([common.augment(self.X[j], self.rng) for j in idx])
        return batch[..., None].astype(np.float32), self.y[idx]

    def on_epoch_end(self):
        self.order = self.rng.permutation(len(self.X))


# ----------------------------------------------------------------- model ---
def build_model():
    """Small CNN, 96x96x1 input with pixel values 0-255 (rescaled inside)."""
    inputs = keras.Input((config.IMG_SIZE, config.IMG_SIZE, 1))
    x = layers.Rescaling(1 / 255.0)(inputs)
    for filters in (16, 32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization(momentum=0.9)(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def convert_tflite(model, X_train, name):
    """Write float32 and full-integer (INT8, uint8 in/out) TFLite models."""
    def make_converter():
        return tf.lite.TFLiteConverter.from_keras_model(model)

    common.model_path(name, "float").write_bytes(make_converter().convert())

    rng = np.random.default_rng(config.SEED)
    idx = rng.choice(len(X_train), min(config.REPRESENTATIVE_SAMPLES, len(X_train)), replace=False)

    def representative():
        for j in idx:                      # half clean, half augmented samples
            img = X_train[j] if j % 2 else common.augment(X_train[j], rng)
            yield [img.reshape(1, config.IMG_SIZE, config.IMG_SIZE, 1).astype(np.float32)]

    conv = make_converter()
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representative
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.uint8
    conv.inference_output_type = tf.uint8
    common.model_path(name, "int8").write_bytes(conv.convert())


# ------------------------------------------------------------------ main ---
def train_one(name):
    print(f"\n===== {name} model: {config.MODELS[name][0]} vs {config.MODELS[name][1]} =====")
    build_cache(name)
    split = common.load_split(name)
    (X_tr, y_tr), (X_val, y_val) = split["train"], split["val"]
    print(f"[{name}] train {len(y_tr)}  val {len(y_val)}  test {len(split['test'][1])}")

    keras.utils.set_random_seed(config.SEED)
    model = build_model()
    history = model.fit(
        AugmentedData(X_tr, y_tr, config.SEED),
        validation_data=(X_val[..., None].astype(np.float32), y_val),
        epochs=config.EPOCHS,
        verbose=2,
        callbacks=[
            # no stopping / LR cuts during warm-up: early validation scores are
            # unreliable until the batch-norm statistics settle
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=config.EARLY_STOP_PATIENCE,
                                          start_from_epoch=config.WARMUP_EPOCHS,
                                          restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6,
                                              cooldown=config.WARMUP_EPOCHS // 2),
        ],
    )
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    model.save(common.model_path(name, "keras"))
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (config.RESULTS_DIR / f"{name}_history.json").write_text(json.dumps(hist))
    convert_tflite(model, X_tr, name)
    print(f"[{name}] saved {common.model_path(name, 'keras').name}, "
          f"{common.model_path(name, 'float').name}, {common.model_path(name, 'int8').name}")


if __name__ == "__main__":
    for model_name in (sys.argv[1:] or list(config.MODELS)):
        train_one(model_name)
