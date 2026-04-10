import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PREPROCESSED_DIRS = [
    'training/preprocessed_data_forwards',
    'training/preprocessed_data_backwards',
    'training/preprocessed_data_forwards_backwards_unstable',
]

BATCH_SIZE = 32
VAL_SPLIT = 0.2

# Load and combine driving_log CSVs from all preprocessed directories
def load_dataset(dirs):
    frames = []
    for d in dirs:
        csv_path = os.path.join(d, 'driving_log.csv')
        df = pd.read_csv(csv_path, header=None, names=['center', 'steering'])
        frames.append(df)
    return pd.concat(frames, ignore_index=True)

# Yields (X, y) batches indefinitely.
# X shape: (batch_size, 66, 200, 3)  float32, normalized to [0, 1]
# y shape: (batch_size,)              float32 steering angles
def batch_generator(samples, batch_size=BATCH_SIZE, training=True):
    n = len(samples)
    indices = np.arange(n)

    while True:
        if training:
            np.random.shuffle(indices)

        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            images = []
            steerings = []

            for i in batch_idx:
                row = samples.iloc[i]
                img = cv2.imread(row['center'].strip())
                if img is None:
                    continue
                img = img.astype(np.float32) / 255.0
                images.append(img)
                steerings.append(float(row['steering']))

            if images:
                yield np.array(images), np.array(steerings, dtype=np.float32)

# Load the full dataset, split into train/val, and return:
# train_gen, val_gen, n_train, n_val
def get_generators(batch_size=BATCH_SIZE):
    df = load_dataset(PREPROCESSED_DIRS)
    train_df, val_df = train_test_split(df, test_size=VAL_SPLIT, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_gen = batch_generator(train_df, batch_size=batch_size, training=True)
    val_gen = batch_generator(val_df, batch_size=batch_size, training=False)

    return train_gen, val_gen, len(train_df), len(val_df)


if __name__ == '__main__':
    train_gen, val_gen, n_train, n_val = get_generators()
    print(f'Training samples : {n_train}')
    print(f'Validation samples: {n_val}')

    X, y = next(train_gen)
    print(f'Batch shape — X: {X.shape}, y: {y.shape}')
    print(f'Pixel range  — min: {X.min():.3f}, max: {X.max():.3f}')
    print(f'Steering range — min: {y.min():.4f}, max: {y.max():.4f}')
