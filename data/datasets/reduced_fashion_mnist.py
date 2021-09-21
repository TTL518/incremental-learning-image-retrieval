from torch import  logical_not, randint
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import time

class ReducedFashionMNIST(Dataset):
    def __init__(self, root="data", train=True, transform=ToTensor(), target_transform=None, classes_to_use=[]):
        
        fashioionMnist = FashionMNIST(root, train, transform, target_transform, True)
        self.data = fashioionMnist.data
        self.targets = fashioionMnist.targets
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.total_batch_time = 0.0
        self.classes_to_use = torch.tensor(classes_to_use)

        mask = torch.any(torch.stack([torch.eq(self.targets, elem).logical_or_(torch.eq(self.targets, elem)) for elem in self.classes_to_use], dim=0), dim = 0)
        self.data = self.data[mask]
        self.targets = self.targets[mask]

        if(train):
            print("Training dataset shape", self.data.shape)
            print("Total classes: ", self.targets.unique())
        else:
            print("Test dataset shape", self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx].view((1,28,28)).type(torch.float)
        label = self.targets[idx]
        return img, label

if __name__ == "__main__":
    rfdataset_train =  ReducedFashionMNIST(train=True, classes_to_use=[0,1,2])
    rfdataset_test =  ReducedFashionMNIST( train=False, classes_to_use=[0,1,2])

    fashionDataset = FashionMNIST("data", train=False, download=True, transform=transforms.ToTensor())
    
    loader = DataLoader(rfdataset_train, 2)

    examples = enumerate(loader)
    batch_idx, (anchor_img, anchor_label) = next(examples)
    #batch_idx, (anchor_img, anchor_label) = next(examples)
    print(batch_idx)
    print(anchor_img.shape)
    print(anchor_label)
