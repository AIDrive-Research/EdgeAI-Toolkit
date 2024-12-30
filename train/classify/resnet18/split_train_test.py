import os
import shutil
import random


def main():
    src_path = 'F:/code/edgeai-toolkit/train/classify/resnet18/dataset/images'
    train_path = 'F:/code/edgeai-toolkit/train/classify/resnet18/dataset/train'
    test_path = 'F:/code/edgeai-toolkit/train/classify/resnet18/dataset/test'

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    labels = os.listdir(src_path)
    for label in labels:
        images_path = os.path.join(src_path, label)
        images = os.listdir(images_path)
        for image in images:
            image_path = os.path.join(images_path, image)
            seed = random.random()
            if seed < 0.2:
                dst_path = os.path.join(test_path, label)
                os.makedirs(dst_path, exist_ok=True)
                shutil.copy(image_path, dst_path)
            else:
                dst_path = os.path.join(train_path, label)
                os.makedirs(dst_path, exist_ok=True)
                shutil.copy(image_path, dst_path)


if __name__ == '__main__':
    main()
