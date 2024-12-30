import os
import argparse
import random
import shutil


def main(args):
    output_images = os.path.join(args.output, 'images')
    output_labels = os.path.join(args.output, 'labels')
    os.makedirs(output_images, exist_ok=True)
    os.makedirs(output_labels, exist_ok=True)
    os.makedirs(os.path.join(output_images, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_images, 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_labels, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_labels, 'val'), exist_ok=True)
    for f in os.listdir(args.input_images):
        seed = random.random()
        if seed < 0.2:
            shutil.move(os.path.join(args.input_images, f), os.path.join(output_images, 'val'))
            shutil.move(os.path.join(args.input_labels, f[:-4] + '.txt'), os.path.join(output_labels, 'val'))
        else:
            shutil.move(os.path.join(args.input_images, f), os.path.join(output_images, 'train'))
            shutil.move(os.path.join(args.input_labels, f[:-4] + '.txt'), os.path.join(output_labels, 'train'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-images', default='voc/JPEGImages/')
    parser.add_argument('--input-labels', default='yolo/labels_temp/')
    parser.add_argument('--output', default='yolo/')
    args = parser.parse_args()
    main(args)
