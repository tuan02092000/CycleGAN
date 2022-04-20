import os
from PIL import Image
from torch.utils.data import Dataset
import config
import numpy as np

class CatDogDataset(Dataset):
    def __init__(self, root_cat, root_dog, transform=None):
        self.root_cat = root_cat
        self.root_dog = root_dog
        self.transform = transform

        self.cat_images = os.listdir(root_cat)
        self.dog_images = os.listdir(root_dog)
        self.length_dataset = max(len(self.cat_images), len(self.dog_images))
        self.cat_len = len(self.cat_images)
        self.dog_len = len(self.dog_images)
    def __len__(self):
        return self.length_dataset
    def __getitem__(self, item):
        cat_img = self.cat_images[item % self.cat_len]
        dog_img = self.dog_images[item % self.dog_len]

        cat_path = os.path.join(self.root_cat, cat_img)
        dog_path = os.path.join(self.root_dog, dog_img)

        cat_img = np.array(Image.open(cat_path).convert("RGB"))
        dog_img = np.array(Image.open(dog_path).convert("RGB"))

        if self.transform:
            augumentations = self.transform(image=cat_img, image0=dog_img)
            cat_img = augumentations["image"]
            dog_img = augumentations["image0"]
        return cat_img, dog_img