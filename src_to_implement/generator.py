import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        
        with open(self.label_path, 'r') as f:
            self.labels = json.load(f)
            
        self.dataset_size = len(self.labels)
        self.keys = [str(i) for i in range(self.dataset_size)]
        
        self.epoch = 0
        self.idx = 0
        
        if self.shuffle:
            np.random.shuffle(self.keys)

    def next(self):
        images = []
        labels = []
        for _ in range(self.batch_size):
            if self.idx >= self.dataset_size:
                self.epoch += 1
                self.idx = 0
                if self.shuffle:
                    np.random.shuffle(self.keys)
                    
            key = self.keys[self.idx]
            self.idx += 1
            
            img = np.load(os.path.join(self.file_path, f"{key}.npy"))
            
            if list(img.shape) != list(self.image_size):
                import scipy.ndimage
                zoom_factors = [t/c for t, c in zip(self.image_size, img.shape)]
                img = scipy.ndimage.zoom(img, zoom_factors)
                
            img = self.augment(img)
            
            images.append(img)
            labels.append(int(self.labels[key]))
            
        return np.array(images), np.array(labels)

    def augment(self, img):
        if self.mirroring:
            if np.random.rand() > 0.5:
                img = np.fliplr(img)
            if np.random.rand() > 0.5:
                img = np.flipud(img)
        if self.rotation:
            k = np.random.choice([1, 2, 3])
            img = np.rot90(img, k)
        return img

    def current_epoch(self):
        return self.epoch

    def class_name(self, x):
        return self.class_dict[x]

    def show(self):
        images, labels = self.next()
        fig, axes = plt.subplots(1, len(images), figsize=(15, 3))
        if len(images) == 1:
            axes = [axes]
        for ax, img, lbl in zip(axes, images, labels):
            ax.imshow(img)
            ax.set_title(self.class_name(lbl))
            ax.axis('off')
        plt.show()

if __name__ == '__main__':
    # Initialize the generator
    gen = ImageGenerator('./exercise_data/', './Labels.json', batch_size=12, image_size=[50, 50, 3], rotation=True, mirroring=True, shuffle=True)
    # Display a batch
    gen.show()