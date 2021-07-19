import torch
import torch.nn as nn
import torch.nn.functional as F

from avalanche.models import IncrementalClassifier

class LeNet_PP(nn.Module):
    """LeNet++ as described in the Center Loss paper."""

    def __init__(self, initial_num_classes):
        super(LeNet_PP, self).__init__()
        self.conv1_1 = nn.Conv2d(1, 32, 5, stride=1, padding=2)
        self.prelu1_1 = nn.PReLU()
        self.conv1_2 = nn.Conv2d(32, 32, 5, stride=1, padding=2)
        self.prelu1_2 = nn.PReLU()
        
        self.conv2_1 = nn.Conv2d(32, 64, 5, stride=1, padding=2)
        self.prelu2_1 = nn.PReLU()
        self.conv2_2 = nn.Conv2d(64, 64, 5, stride=1, padding=2)
        self.prelu2_2 = nn.PReLU()
        
        self.conv3_1 = nn.Conv2d(64, 128, 5, stride=1, padding=2)
        self.prelu3_1 = nn.PReLU()
        self.conv3_2 = nn.Conv2d(128, 128, 5, stride=1, padding=2)
        self.prelu3_2 = nn.PReLU()
        
        self.fc1 = nn.Linear(128*3*3, 2)
        self.prelu_fc1 = nn.PReLU()
        #self.fc2 = nn.Linear(2, num_classes)
        self.inc_classifier = IncrementalClassifier(2, initial_num_classes)
        

    def forward(self, x):
        x = self.prelu1_1(self.conv1_1(x))
        x = self.prelu1_2(self.conv1_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu2_1(self.conv2_1(x))
        x = self.prelu2_2(self.conv2_2(x))
        x = F.max_pool2d(x, 2)
        
        x = self.prelu3_1(self.conv3_1(x))
        x = self.prelu3_2(self.conv3_2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, 128*3*3)
        x = self.prelu_fc1(self.fc1(x))
        y = self.inc_classifier(x)
        
        return x, y

def test():
    net = LeNet_PP(2)
    features, y = net(torch.randn(1, 1, 28, 28))
    print(y.size())
    print(features.size())


if __name__ == "__main__":
    test()