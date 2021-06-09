from torch import  logical_not, randint
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from PIL import Image

class TripletFashionMnist(Dataset):
    def __init__(self, train=True, transform=ToTensor(), target_transform=None):
        
        fashioionMnist = FashionMNIST("data", train, transform, target_transform, True)
        self.data = fashioionMnist.data
        self.targets = fashioionMnist.targets
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        anchor_img = self.data[idx]
        anchor_label = self.targets[idx]

        if self.train:
            mask_pos = self.targets.eq(anchor_label)
            positive_img_list = self.data[mask_pos,:,:]
            positive_random_idx = randint(0, positive_img_list.size(0),(1,))
            positive_img = positive_img_list[positive_random_idx].view((28,28))

            mask_neg = logical_not(mask_pos)
            negative_img_list = self.data[mask_neg, :, :]
            negative_random_idx = randint(0, negative_img_list.size(0),(1,))
            negative_img = negative_img_list[negative_random_idx].view((28,28))

            #print(anchor_img.shape)
            #print(negative_img.shape)
            #print(positive_img.shape)

            anchor_img = Image.fromarray(anchor_img.numpy(), mode='L')
            positive_img = Image.fromarray(positive_img.numpy(), mode='L')
            negative_img = Image.fromarray(negative_img.numpy(), mode='L')

            if self.transform is not None:
                anchor_img = self.transform(anchor_img)
                positive_img = self.transform(positive_img)
                negative_img = self.transform(negative_img)
            
            
            return anchor_img, anchor_label, positive_img, negative_img
        else:
            if self.transform is not None:
                anchor_img = self.transform(anchor_img)
          
            return anchor_img, anchor_label

if __name__ == "__main__":
    tfdataset_train =  TripletFashionMnist(train=True)
    tfdataset_test =  TripletFashionMnist( train=False)

    fashionDataset = FashionMNIST("data", train=True, download=True, transform=transforms.ToTensor())
    
    loader = DataLoader(tfdataset_train, 64)

    examples = enumerate(loader)
    batch_idx, (anchor_img, anchor_label, positive_img, negative_img) = next(examples)
    print(batch_idx)
    print(anchor_img[0].shape)

