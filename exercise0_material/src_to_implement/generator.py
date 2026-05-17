import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform

CLASS_NAMES = {
    0: 'airplane', 1: 'automobile', 2: 'bird',  3: 'cat',  4: 'deer',
    5: 'dog',      6: 'frog',       7: 'horse', 8: 'ship', 9: 'truck',
}

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring

        self.shuffle = shuffle
        with open(label_path) as f:
            self.labels = json.load(f)

        self.file_names = list(self.labels.keys())
        if shuffle:
            np.random.shuffle(self.file_names)

        self.current_index = 0
        self.epoch = 0
        self.image_paths = [os.path.join(file_path, f) for f in os.listdir(file_path) if f.endswith('.npy')]

    def next(self):
        collected_images = []
        collected_labels = []

        for _ in range(self.batch_size):
            if self.current_index >= len(self.file_names):
                self.epoch += 1
                self.current_index = 0
                if self.shuffle:
                    np.random.shuffle(self.file_names)

            file_name = self.file_names[self.current_index]
            img = np.load(os.path.join(self.file_path, f"{file_name}.npy"))

            img = transform.resize(img, self.image_size)
            img = self.augment(img)

            collected_images.append(img)
            collected_labels.append(self.labels[file_name])

            self.current_index += 1

        return np.array(collected_images), np.array(collected_labels)

    def augment(self, img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if self.mirroring and np.random.rand() < 0.5:
            img = np.fliplr(img)
        if self.rotation:
            k = np.random.randint(1, 4)   # 90, 180, or 270 degrees
            img = np.rot90(img, k=k)
        return img

    def current_epoch(self):
        return self.epoch

    def class_name(self, x):
        # This function returns the class name for a specific input
        return CLASS_NAMES[x]

    def show(self):
        images, labels = self.next()
        for i, (img, label) in enumerate(zip(images, labels)):
            plt.subplot(2, self.batch_size // 2, i + 1)
            plt.imshow(img)
            plt.title(self.class_name(label))
            plt.axis('off')
        plt.show()
