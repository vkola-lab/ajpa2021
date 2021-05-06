import os
import torch.utils.data as data
import numpy as np
from PIL import Image, ImageFile
import random
from torchvision.transforms import ToTensor
from torchvision import transforms

class KidneyFibrosis(data.Dataset):
    """input and label image dataset"""

    def __init__(self, root, ids, label=False, classdict={}, transform=False):
        super(KidneyFibrosis, self).__init__()
        """
        Args:

        fileDir(string):  directory with all the input images.
        transform(callable, optional): Optional transform to be applied on a sample
        """
        self.root = root
        self.label = label
        self.transform = transform
        self.ids = ids
        self.classdict = classdict

    def __getitem__(self, index):
        sample = {}
        sample['id'] = self.ids[index].split('.')[0]
        image = Image.open(os.path.join(self.root, self.ids[index]) + '.png').convert('RGB') # w, h
        sample['image'] = image

        if self.label:
            label_info = self.classdict[sample['id']]
            sample['label'] = label_info

        return sample

    def __len__(self):
        return len(self.ids)
