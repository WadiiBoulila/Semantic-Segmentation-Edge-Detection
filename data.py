import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class DataPrep(Dataset):
    def __init__(self, data):


        self.total_data = data
        self.length = len(self.total_data)

    def __len__(self):
        return len(self.total_data)
        #return len(self.total_data)
    
    
    def __getitem__(self, index):
        image, mask, edges, _ = self.total_data[index]
        image = cv2.resize(image, (512, 512))
        image = np.transpose(image, (2, 0, 1))

        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)
        edges = cv2.resize(edges, (512, 512), interpolation=cv2.INTER_NEAREST)

        mask[mask == 255] = 1
        edges[edges == 255] = 1

        mask = torch.Tensor(mask)
        mask = mask.long()
        edges = torch.Tensor(edges)
        edges = edges.long()

        image = image/np.max(image)
        image = torch.Tensor(image)
        
        return image, mask