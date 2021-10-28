from avalanche.benchmarks.datasets import FashionMNIST, CUB200, MNIST, CIFAR100
from data.datasets.reduced_fashion_mnist import ReducedFashionMNIST
from torchvision.transforms import ToTensor
from torchvision.transforms.functional import InterpolationMode
from torchvision import transforms

def get_train_test_dset(dset_name):
    train_dset = None
    test_dset = None
    class_names = None
    
    if(dset_name=="FashionMNIST"):

        train_dset = FashionMNIST(root="data", train=True, transform=ToTensor())
        test_dset = FashionMNIST(root="data", train=False, transform=ToTensor())

        class_names = {
            0:"T-shirt",
            1:"Trousers",
            2:"Pullover",
            3:"Dress",
            4:"Coat",
            5:"Sandal",
            6:"Shirt",
            7:"Sneaker",
            8:"Bag",
            9:"Ankle boot"
        }

    elif(dset_name=="ReducedFashionMNIST"):
        train_dset = ReducedFashionMNIST(root="data", train=True, transform=ToTensor(), classes_to_use=[0,1])
        test_dset = ReducedFashionMNIST(root="data", train=False, transform=ToTensor(), classes_to_use=[0,1])
    
        class_names = {
            0:"T-shirt",
            1:"Trousers",
            2:"Pullover",
            3:"Dress",
            4:"Coat",
            5:"Sandal",
            6:"Shirt",
            7:"Sneaker",
            8:"Bag",
            9:"Ankle boot"
        }
    
    elif(dset_name=="MNIST"):
        
        train_dset = MNIST(root="data", train=True, transform=ToTensor())
        test_dset = MNIST(root="data", train=False, transform=ToTensor())

        class_names = {
            0:"0",
            1:"1",
            2:"2",
            3:"3",
            4:"4",
            5:"5",
            6:"6",
            7:"7",
            8:"8",
            9:"9"
        }

    elif(dset_name=="CUB200"):
        
        input_size = (224,224)
        train_transform = transforms.Compose([transforms.Resize(input_size, InterpolationMode.BILINEAR),
                                        transforms.CenterCrop(input_size),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        test_transform = transforms.Compose([transforms.Resize(input_size, InterpolationMode.BILINEAR),
                                        transforms.CenterCrop(input_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        # train_transform=transforms.Compose([transforms.Resize((400, 400), InterpolationMode.BILINEAR),
        #                             transforms.RandomCrop((256, 256)),
        #                             transforms.RandomHorizontalFlip(),
        #                             transforms.ToTensor(),
        #                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # test_transform=transforms.Compose([transforms.Resize((400, 400), InterpolationMode.BILINEAR),
        #                             transforms.CenterCrop((256, 256)),
        #                             transforms.ToTensor(),
        #                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        train_dset = CUB200(root="data", train=True, transform=train_transform, download=True)
        test_dset = CUB200(root="data", train=False, transform=test_transform, download=True)
        class_names = None
    
    elif(dset_name=="CIFAR100"):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_dset = CIFAR100(root="data", train=True, download=True, transform=transform_train)
        test_dset = CIFAR100(root="data", train=False, download=True, transform=transform_test)
        class_names = None

    else: 
        raise ValueError("Dataset name isn't correct")
    
    return train_dset, test_dset, class_names

if __name__ == "__main__":
    train_dset=CIFAR100(root="data", train=True, transform=ToTensor())
    train_dset=CIFAR100(root="data", train=False, transform=ToTensor())
    class_names = None
