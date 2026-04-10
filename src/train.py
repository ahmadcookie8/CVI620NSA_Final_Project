import math
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from batch_data import get_generators, BATCH_SIZE

# Hyperparameters
EPOCHS     = 1000  # run until loss plateaus with a low learning rate
LR         = 1e-3
BATCH_SIZE = BATCH_SIZE   # inherited from batch_data - 32
MODEL_PATH = 'models/model.h5'


# CNN matching layers in pdf instructions
def build_model(input_shape=(66, 200, 3)):
    model = Sequential([
        # 5 convolutional layers
        Conv2D(24, (5, 5), strides=(2, 2), activation='relu', input_shape=input_shape),
        Conv2D(36, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(48, (5, 5), strides=(2, 2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),

        Flatten(),

        # 3 fully-connected layers
        Dense(100, activation='relu'),
        Dropout(0.2), # dropout to remove some neurons to prevent generalization and "memorization"
        Dense(50, activation='relu'),
        Dropout(0.2),
        Dense(10, activation='relu'),

        # steering angle
        Dense(1),
    ])
    return model


# Training
def train():
    train_gen, val_gen, n_train, n_val = get_generators(batch_size=BATCH_SIZE)

    steps_per_epoch  = math.ceil(n_train / BATCH_SIZE)
    validation_steps = math.ceil(n_val   / BATCH_SIZE)

    model = build_model()
    model.summary()

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss='mse',
    )

    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping( # stop early if loss hasn't improved in 10 epochs
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau( # halves the learning rate if no improvement in 5 epochs (until min learning rate)
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    plot_training(history)
    print(f'\nBest model saved to {MODEL_PATH}')


# Plot training graphs
def plot_training(history):
    train_loss = history.history['loss']
    val_loss   = history.history['val_loss']
    epochs     = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label='Training loss')
    plt.plot(epochs, val_loss,   label='Validation loss')
    plt.title('Training vs Validation Loss (MSE)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('docs/training_loss.png', dpi=150)
    plt.show()
    print('Loss plot saved to docs/training_loss.png')


if __name__ == '__main__':
    train()
