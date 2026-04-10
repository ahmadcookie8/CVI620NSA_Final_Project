import os
import cv2
import pandas as pd

# vertical crop rows matching the nvidia model input region
CROP_TOP = 60
CROP_BOTTOM = 135

# target size used by the nvidia model
TARGET_W = 200
TARGET_H = 66

DATASETS = [
    {
        'src': 'training/augmented_data_forwards',
        'dst': 'training/preprocessed_data_forwards',
    },
    {
        'src': 'training/augmented_data_backwards',
        'dst': 'training/preprocessed_data_backwards',
    },
    {
        'src': 'training/augmented_data_forwards_backwards_unstable',
        'dst': 'training/preprocessed_data_forwards_backwards_unstable'
    },
]


def preprocess(img):
    img = img[CROP_TOP:CROP_BOTTOM, :, :]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (TARGET_W, TARGET_H))
    # normalization is left to the training script at load time because PNG cannot store float values..
    return img


def process_dataset(src_dir, dst_dir):
    dst_img_dir = os.path.join(dst_dir, 'IMG')
    os.makedirs(dst_img_dir, exist_ok=True)

    csv_path = os.path.join(src_dir, 'driving_log.csv')
    df = pd.read_csv(csv_path, header=None, names=['center', 'steering'])

    new_rows = []

    for _, row in df.iterrows():
        src_img_path = row['center'].strip()
        steering = float(row['steering'])

        img = cv2.imread(src_img_path)
        if img is None:
            print(f'  WARNING: could not read {src_img_path}, skipping.')
            continue

        img = preprocess(img)

        filename = os.path.basename(src_img_path)
        dst_img_path = os.path.join(dst_img_dir, filename)
        cv2.imwrite(dst_img_path, img)

        csv_img_path = dst_dir + '/IMG/' + filename
        new_rows.append([csv_img_path, steering])

    new_df = pd.DataFrame(new_rows)
    new_df.to_csv(os.path.join(dst_dir, 'driving_log.csv'), header=False, index=False)

    print(f'  {len(new_rows)} images written to {dst_dir}/')


def main():
    for dataset in DATASETS:
        src = dataset['src']
        dst = dataset['dst']
        print(f'Processing {src} -> {dst} ...')
        process_dataset(src, dst)
    print('Done.')


if __name__ == '__main__':
    main()
