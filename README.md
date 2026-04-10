# CVI620NSA Final Project - Self-Driving Car Simulation

A self-driving car simulation using an NVIDIA end-to-end CNN trained on data collected from the Udacity self-driving car simulator.

## Project Structure

```
src/                        # Source code
    train.py                    # Model training script
    batch_data.py               # Dataset loader and batch generator
    augment_data.py             # Data augmentation pipeline
    preprocess_data.py          # Image preprocessing pipeline
    TestSimulation.py           # Simulator inference server
    fix_csv_paths.py            # Utility to fix CSV image paths
    histogram/
        plot_steering_histogram.py  # Steering angle distribution plot

training/                   # All training data
    training_data_forwards/             # Raw data of forward laps
    training_data_backwards/            # Raw data of backward laps
    training_data_forwards_backwards_unstable/  # Raw data of recovery driving
    augmented_data_forwards/            # Augmented forward data
    augmented_data_backwards/           # Augmented backward data
    augmented_data_forwards_backwards_unstable/ # Augmented recovery data
    preprocessed_data_forwards/         # Preprocessed forward data
    preprocessed_data_backwards/        # Preprocessed backward data
    preprocessed_data_forwards_backwards_unstable/ # Preprocessed recovery data

models/                     # Saved model weights
    model.h5

docs/                       # Documentation and media
    Final_Project.pdf
    training_loss.png
    steering_histogram.png
    Self-Driving-Demo.mp4

README.md
package_list.txt
```

## Pipeline

1. **Collect** - Record driving data in the Udacity simulator (saved to `training/training_data_*/`)
2. **Augment** - Run `src/augment_data.py` to apply flip, brightness, zoom, pan, and rotate augmentations
3. **Preprocess** - Run `src/preprocess_data.py` to crop, convert to YUV, blur, and resize images to 200×66
4. **Train** - Run `src/train.py` to train the NVIDIA CNN; best model saved to `models/model.h5`
5. **Test** - Run `src/TestSimulation.py`, then launch the Udacity simulator in autonomous mode

## Setup

Install dependencies into a Python virtual environment:

```bash
pip install -r package_list.txt
```

## Running

All scripts are run from the project root:

```bash
# Augment raw training data
python src/augment_data.py

# Preprocess augmented data
python src/preprocess_data.py

# Train the model
python src/train.py

# Start the inference server
python src/TestSimulation.py

# Open the driving simulator and start autonomous mode
```
