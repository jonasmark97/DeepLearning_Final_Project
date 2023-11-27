import os
import numpy as np
import glob
import PIL.Image as Image

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

NUMBER_OF_LABELS = 3


# Classification dataset - Evenly distributed amongst the label

# Multispectral
class MaizeBlobDataset(torch.utils.data.Dataset):
    def __init__(self, split, validation_ratio, testing_ratio, transform=transforms.ToTensor() ,random_seed=2):

        self.transform = transform
        self.split = split

        base_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)),"Data/blobs_multispectral_labelled_uniform_dist")
        N = len(os.listdir(base_dir)) // NUMBER_OF_LABELS
 
        self.split = split
        np.random.seed(random_seed)
        indexes = np.arange(N, dtype=int)
        np.random.shuffle(indexes)
        if self.split=="train":
            # first x percentage
            i_start = 0
            i_stop = int(N*(1-validation_ratio-testing_ratio))

        elif self.split=="valid":
            # between train and test percentage 
            i_start = int(N*(1-validation_ratio-testing_ratio))
            i_stop = i_start + int(N*validation_ratio)

        elif self.split=="test":
            # after train + validation percentage
            i_start = int(N*(1-validation_ratio-testing_ratio)) + int(N*validation_ratio)
            i_stop = N
        
        else:
            raise SyntaxError("split is only [\"train\",\"valid\",\"test\"]")
                
        n = (i_stop - i_start) * NUMBER_OF_LABELS
        self.labels = np.zeros(n,dtype=int)
        self.image_paths = ["" for _ in range(n)]
        index = 0
        for j, label in enumerate(["Background", "MaizGood", "MaizBad"]):
            for i in range(i_stop - i_start):
                self.labels[index] = j
                self.image_paths[index] = os.path.join(base_dir, label + "_" + str(i+1) + ".npy")

                index+=1




    def __len__(self):
        'Returns the total number of samples'
        return len(self.image_paths)
    

    def __getitem__(self, idx):
        'Generates one sample of data'
        X = np.load(self.image_paths[idx]) / 55.0  # 55.0 is the max number in multispectral images from Videometer
        Y = self.labels[idx]
        X = self.transform(X)

        return X, Y
        
    
    
            
def get_data_blobs(batch_size, validation_ratio, testing_ratio, train_transform=transforms.ToTensor(), general_transform=transforms.ToTensor(), n_patches_per_image=1):   

    trainset = MaizeBlobDataset(split="train", validation_ratio=validation_ratio, testing_ratio=testing_ratio, transform=train_transform,random_seed=2)
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=3)
    validset = MaizeBlobDataset(split="valid", validation_ratio=validation_ratio, testing_ratio=testing_ratio,  transform=general_transform, random_seed=2)
    valid_loader = DataLoader(validset, batch_size=batch_size, num_workers=3)
    testset = MaizeBlobDataset(split="test", validation_ratio=validation_ratio, testing_ratio=testing_ratio, transform=general_transform, random_seed=2)
    test_loader = DataLoader(testset, batch_size=1, num_workers=3)

    
    print('Loaded %d training images' % len(trainset))
    print('Loaded %d validation images' % len(validset))
    print('Loaded %d test images' % len(testset))

    return train_loader, valid_loader, test_loader


# train_loader, valid_loader, test_loader = get_data_blobs(8, validation_ratio=0.15, testing_ratio=0.1)




