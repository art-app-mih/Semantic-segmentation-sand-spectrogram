import os
import numpy as np 
import cv2
import torch
import torch.utils.data as data


class DataLoaderSegmentation(data.Dataset):

    def __init__(self, folder_path, images_folder, mask_folder, transform=None):
        super(DataLoaderSegmentation, self).__init__()
        self.images_path = os.path.join(folder_path, images_folder)
        self.img_files = os.listdir(self.images_path)
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(folder_path, mask_folder, os.path.basename(img_path)))
        self.transform = transform

    def __getitem__(self, index):
            img_path = self.img_files[index]
            mask_path = self.mask_files[index]
            data = cv2.imread(os.path.join(self.images_path, img_path))
            label = cv2.imread(mask_path)
            #Normalize data [255, 0] -> [1, 0]
            for i in label:
                for x in i:
                    for g in range(len(x)):
                        if x[g] > 0:
                            x[g] = 1
                        else:
                            x[g] = 0
            if self.transform is not None:
                data = self.transform(data)
                label = self.transform(label)
            
            label = torch.from_numpy(label)
            mask = torch.empty(512, 512, dtype=torch.long)

            colors = torch.unique(label.reshape(-1, label.size(2)), dim=0).numpy()
            label = label.permute(2, 0, 1).contiguous()
            mapping = {tuple(c): t for c, t in zip(colors.tolist(), range(len(colors)))}

            for k in mapping:
                # Get all indices for current class
                idx = (label==torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
                validx = (idx.sum(0) == 3)  # Check that all channels match
                mask[validx] = torch.tensor(mapping[k], dtype=torch.long)
                
            return torch.from_numpy(data).type(torch.float32).permute(2, 0, 1), mask

    def __len__(self):
        return len(self.img_files)
