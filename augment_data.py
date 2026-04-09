import os
import random
import cv2
import numpy as np
import pandas as pd

AUGMENTATION_PROB = 0.25

DATASETS = [
    # {
    #     'src': 'training_data_forwards',
    #     'dst': 'augmented_data_forwards',
    # },
    # {
    #     'src': 'training_data_backwards',
    #     'dst': 'augmented_data_backwards',
    # },
    {
        'src': 'training_data_forwards_backwards_unstable',
        'dst': 'augmented_data_forwards_backwards_unstable'
    },
]


def aug_flip(img, steering):
    return cv2.flip(img, 1), steering * -1.0


def aug_brightness(img, steering):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    factor = random.uniform(0.4, 1.2)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR), steering


def aug_zoom(img, steering):
    h, w = img.shape[:2]
    zoom = random.uniform(1.1, 1.3)
    new_h = int(h / zoom)
    new_w = int(w / zoom)
    y1 = (h - new_h) // 2
    x1 = (w - new_w) // 2
    cropped = img[y1:y1 + new_h, x1:x1 + new_w]
    return cv2.resize(cropped, (w, h)), steering


def aug_pan(img, steering):
    h, w = img.shape[:2]
    tx = random.uniform(-0.2, 0.2) * w
    ty = random.uniform(-0.1, 0.1) * h
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    panned = cv2.warpAffine(img, M, (w, h))
    # small steering correction proportional to horizontal shift
    new_steering = steering + (tx / w) * 0.2
    return panned, new_steering


def aug_rotate(img, steering):
    h, w = img.shape[:2]
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated, steering


AUGMENTATIONS = [aug_flip, aug_brightness, aug_zoom, aug_pan, aug_rotate]


def process_dataset(src_dir, dst_dir):
    dst_img_dir = os.path.join(dst_dir, 'IMG')
    os.makedirs(dst_img_dir, exist_ok=True)

    csv_path = os.path.join(src_dir, 'driving_log.csv')
    df = pd.read_csv(csv_path, header=None, names=['center', 'steering'])

    augmented_count = 0
    new_rows = []

    for _, row in df.iterrows():
        src_img_path = row['center'].strip()
        steering = float(row['steering'])

        img = cv2.imread(src_img_path)
        if img is None:
            print(f'  WARNING: could not read {src_img_path}, skipping.')
            continue

        # Always write the original (clean) image
        filename = os.path.basename(src_img_path)
        dst_img_path = os.path.join(dst_img_dir, filename)
        cv2.imwrite(dst_img_path, img)
        new_rows.append([dst_dir + '/IMG/' + filename, steering])

        # Additionally, augment 25% of images and write as a separate file
        if random.random() < AUGMENTATION_PROB:
            aug_fn = random.choice(AUGMENTATIONS)
            aug_img, aug_steering = aug_fn(img, steering)
            aug_steering = float(np.clip(aug_steering, -1.0, 1.0))
            augmented_count += 1

            name, ext = os.path.splitext(filename)
            aug_filename = name + '_aug' + ext
            aug_img_path = os.path.join(dst_img_dir, aug_filename)
            cv2.imwrite(aug_img_path, aug_img)
            new_rows.append([dst_dir + '/IMG/' + aug_filename, aug_steering])

    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(os.path.join(dst_dir, 'driving_log.csv'), header=False, index=False)

    total = len(new_rows)
    print(f'  {total} images written ({augmented_count} augmented, {total - augmented_count} unchanged)')


def main():
    for dataset in DATASETS:
        src = dataset['src']
        dst = dataset['dst']
        print(f'Processing {src} -> {dst} ...')
        process_dataset(src, dst)
    print('Done.')


if __name__ == '__main__':
    main()
