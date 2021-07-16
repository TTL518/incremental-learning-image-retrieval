import torch 
from torch import  logical_not, randint
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from tqdm import tqdm

from avalanche.benchmarks.datasets import CUB200

class TripletCUB200(Dataset):
    def __init__(self, train=True, transform=ToTensor(), target_transform=None):
        
        cub200 = CUB200(root="data", train=train, transform=transform, target_transform=target_transform, download=True)
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.targets = torch.tensor((cub200.targets))
        self.data = torch.zeros((len(cub200._images),cub200[0][0].shape[0], cub200[0][0].shape[1], cub200[0][0].shape[2]))

        print("Caricamento dataset")
        for i in tqdm(range(len(cub200.targets))):
            self.data[i] = cub200[i][0]

        if(train):
            print("Training dataset shape", self.data.shape)
        else:
            print("Test dataset shape", self.targets.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_img = self.data[idx]
        anchor_label = self.targets[idx]
        #print("Data shape: ", self.data.shape)

        if self.train:
            mask_pos = self.targets.eq(anchor_label)
            positive_img_list = self.data[mask_pos,:,:]
            positive_random_idx = randint(0, positive_img_list.size(0),(1,))
            positive_img = positive_img_list[positive_random_idx].view((self.data.shape[1], self.data.shape[2], self.data.shape[3]))

            mask_neg = logical_not(mask_pos)
            negative_img_list = self.data[mask_neg, :, :]
            negative_random_idx = randint(0, negative_img_list.size(0),(1,))
            negative_img = negative_img_list[negative_random_idx].view((self.data.shape[1], self.data.shape[2], self.data.shape[3]))

            if len(anchor_img.shape) == 2:
                anchor_img = np.stack([anchor_img] * 3, 2)
            
            return anchor_img, anchor_label , positive_img, negative_img
        else:
            return anchor_img, anchor_label

if __name__ == "__main__":
    
    train_transform=transforms.Compose([transforms.Resize((600, 600), InterpolationMode.BILINEAR),
                                    transforms.RandomCrop((448, 448)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform=transforms.Compose([transforms.Resize((600, 600), InterpolationMode.BILINEAR),
                                    transforms.CenterCrop((448, 448)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    cub = TripletCUB200(train=False, transform=train_transform)
    
    loader = DataLoader(cub, 2)

    examples = enumerate(loader)
    batch_idx, (anchor_img, anchor_label) = next(examples)
    #batch_idx, (anchor_img, anchor_label, positive_img, negative_img) = next(examples)

    print(batch_idx)
    print(anchor_img.shape)
