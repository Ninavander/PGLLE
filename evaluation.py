import tensorflow as tf
import numpy as np
import cv2
import os

def psnr(img1, img2):
    return tf.reduce_mean(tf.image.psnr(img1, img2, 1.0))

def ssim(img1, img2):
    return tf.reduce_mean(tf.image.ssim(img1, img2, 1.0))

def load_image_pairs(image_dir1, image_dir2):
    file_list1 = sorted([f for f in os.listdir(image_dir1) if f.endswith('.png')])
    file_list2 = sorted([f for f in os.listdir(image_dir2) if f.endswith('.png')])

    prefixes1 = {f.split('_')[0]: [] for f in file_list1}
    prefixes2 = {f.split('_')[0]: [] for f in file_list2}

    for f in file_list1:
        prefix = f.split('_')[0]
        suffix = f.split('_')[1].replace('.png', '')
        if prefix in prefixes1:
            prefixes1[prefix].append(os.path.join(image_dir1, f))

    for f in file_list2:
        prefix = f.split('_')[0]
        suffix = f.split('_')[1].replace('.png', '')
        if prefix in prefixes2:
            prefixes2[prefix].append(os.path.join(image_dir2, f))

    common_prefixes = set(prefixes1.keys()).intersection(set(prefixes2.keys()))

    image_pairs = {}

    for prefix in common_prefixes:
        sorted_images1 = sorted(prefixes1[prefix])
        sorted_images2 = sorted(prefixes2[prefix])

        if len(sorted_images1) >= 3 and len(sorted_images2) >= 3:
            image_pairs[prefix] = {
                'aop': (sorted_images1[0], sorted_images2[0]),
                'dop': (sorted_images1[1], sorted_images2[1]),
                's0': (sorted_images1[2], sorted_images2[2]),
            }
        else:
            print(f"Not enough images for prefix {prefix} in one of the directories.")

    # print("Image pairs:", image_pairs)
    return image_pairs
# 'indoor011': {'aop': ('PILE/test/output/our_sad/indoor011_6000_aop.png', 'PILE/test/output/label/indoor011_60000_aop.png'), 'dop': ('PILE/test/output/our_sad/indoor011_6000_dolp.png', 'PILE/test/output/label/indoor011_60000_dolp.png'), 's0': ('PILE/test/output/our_sad/indoor011_6000_s0.png', 'PILE/test/output/label/indoor011_60000_s0.png')}}

def calculate_metrics(image_dir1, image_dir2):
    metrics = {'psnr': {'s0': [], 'aop': [], 'dop': []}, 'ssim': {'s0': [], 'aop': [], 'dop': []}}
    image_pairs = load_image_pairs(image_dir1, image_dir2)

    with tf.Session() as sess:
        for prefix, images in image_pairs.items():
            for suffix in ['s0', 'aop', 'dop']:
                img1_path, img2_path = images[suffix]

               
                img1 = cv2.imread(img1_path, cv2.IMREAD_COLOR)
                img2 = cv2.imread(img2_path, cv2.IMREAD_COLOR)

                # # Resize img2 to match the dimensions of img1
                # img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
                img1 = img1.astype(np.float32) / 255.0
                img2 = img2.astype(np.float32) / 255.0

                img1_tensor = tf.convert_to_tensor(img1)
                img2_tensor = tf.convert_to_tensor(img2)

                if img1 is None or img2 is None:
                    print(f"Error reading images: {img1_path}, {img2_path}")
                    continue

                psnr_value = sess.run(psnr(img1_tensor, img2_tensor))
                ssim_value = sess.run(ssim(img1_tensor, img2_tensor))

                metrics['psnr'][suffix].append(psnr_value)
                metrics['ssim'][suffix].append(ssim_value)

                print(f"Prefix: {prefix}, Suffix: {suffix}")
                print(f"  PSNR: {psnr_value:.4f}")
                print(f"  SSIM: {ssim_value:.4f}")

        average_metrics = {}
        for suffix in ['s0', 'aop', 'dop']:
            average_psnr = np.mean(metrics['psnr'][suffix])
            average_ssim = np.mean(metrics['ssim'][suffix])
            average_metrics[suffix] = {'average_psnr': average_psnr, 'average_ssim': average_ssim}

            print(f"\nAverage Metrics for {suffix}:")
            print(f"  Average PSNR: {average_psnr:.4f}")
            print(f"  Average SSIM: {average_ssim:.4f}")

    return metrics, average_metrics

# Example usage
our_sad_dir = 'PILE/test/output_color2/our_sad'
label_dir = 'PILE/test/output_color2/label'

metrics, average_metrics = calculate_metrics(our_sad_dir, label_dir)



