import torch
import torch.nn as nn
import torch.nn.functional as F

from models.NewIncrementalClassifier import NewIncrementalClassifier



class LeNet_PP(nn.Module):
    """LeNet++ as described in the Center Loss paper."""

    def __init__(self, initial_num_classes, bias_classifier, norm_classifier):
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
        self.inc_classifier = NewIncrementalClassifier(2, initial_num_classes, bias_classifier, norm_classifier)
        

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
        
        x2 = x.view(-1, 128*3*3)
        x1 = self.prelu_fc1(self.fc1(x2))
        y = self.inc_classifier(x1)

        return y, x1, x2

class LeNet(nn.Module):

    def __init__(self, initial_num_classes, bias_classifier, norm_classifier):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 4*4 from image dimension (28x28)
        self.fc2 = nn.Linear(120, 84)
        #self.fc3 = nn.Linear(84, 10)
        self.inc_classifier = NewIncrementalClassifier(84, initial_num_classes, bias_classifier, norm_classifier)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x2 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x2))
        #y = self.fc3(x)
        y = self.inc_classifier(x1)
        return y, x1, x2

def test():
    net = LeNet(2, False, True)
    y, x = net(torch.randn(1, 1, 28, 28))
    print(y.size())
    print(x.size())


if __name__ == "__main__":
    test()