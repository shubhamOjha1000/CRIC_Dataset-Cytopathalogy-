from PIL import Image
import numpy as np
import pandas as pd 
import os
from torch.utils.data import Dataset


class CricDataset(Dataset):
    def __init__(self, img_dir, annotation_file, img_transform = None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(annotation_file)
        self.img_transform = img_transform

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        if self.img_transform is not None:
            image = image.resize((224, 224))
        image = np.array(image)
        image = np.transpose(image, (2, 0, 1))
        image = image / 255.0
        

        label = self.img_labels.iloc[idx, 1]

        return image, label

        





