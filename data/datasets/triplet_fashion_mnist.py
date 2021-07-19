from torch import  logical_not, randint
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
from torchvision import transforms

class TripletFashionMnist(Dataset):
    def __init__(self, train=True, transform=ToTensor(), target_transform=None):
        
        fashioionMnist = FashionMNIST("data", train, transform, target_transform, True)
        self.data = fashioionMnist.data
        self.targets = fashioionMnist.targets
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        if(train):
            print("Training dataset shape", self.data.shape)
        else:
            print("Test dataset shape", self.data.shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_img = self.data[idx].view((1,28,28))
        anchor_img = anchor_img.type(torch.float)
        anchor_label = self.targets[idx]

        if self.train:
            mask_pos = self.targets.eq(anchor_label)
            positive_img_list = self.data[mask_pos,:,:]
            positive_random_idx = randint(0, positive_img_list.size(0),(1,))
            positive_img = positive_img_list[positive_random_idx].view((1,28,28))

            mask_neg = logical_not(mask_pos)
            negative_img_list = self.data[mask_neg, :, :]
            negative_random_idx = randint(0, negative_img_list.size(0),(1,))
            negative_img = negative_img_list[negative_random_idx].view((1,28,28))

            positive_img = positive_img.type(torch.float)
            negative_img = negative_img.type(torch.float)

            return anchor_img, anchor_label, positive_img, negative_img
        else:
            return anchor_img, anchor_label

if __name__ == "__main__":
    tfdataset_train =  TripletFashionMnist(train=True)
    tfdataset_test =  TripletFashionMnist( train=False)

    fashionDataset = FashionMNIST("data", train=False, download=True, transform=transforms.ToTensor())
    
    loader = DataLoader(tfdataset_train, 2)

    examples = enumerate(loader)
    batch_idx, (anchor_img, anchor_label, positive_img, negative_img) = next(examples)
    #batch_idx, (anchor_img, anchor_label) = next(examples)
    print(batch_idx)
    print(anchor_img.shape)
    print(anchor_label)
