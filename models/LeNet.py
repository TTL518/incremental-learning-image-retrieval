import torch
import torch.nn as nn
import torch.nn.functional as F

from avalanche.models import DynamicModule
from avalanche.benchmarks.utils import AvalancheDataset

class NewIncrementalClassifier(DynamicModule):
    def __init__(self, in_features, initial_out_features=2, bias=True, norm_weights=False):
        """ Output layer that incrementally adds units whenever new classes are
        encountered.

        Typically used in class-incremental benchmarks where the number of
        classes grows over time.

        :param in_features: number of input features.
        :param initial_out_features: initial number of classes (can be
            dynamically expanded).
        """
        super().__init__()
        self.bias = bias
        self.classifier = torch.nn.Linear(in_features, initial_out_features, bias)
        self.norm_weights = norm_weights
        if self.norm_weights:
            self.classifier.weight.data = F.normalize(self.classifier.weight.data) 

    @torch.no_grad()
    def adaptation(self, dataset: AvalancheDataset):
        """ If `dataset` contains unseen classes the classifier is expanded.

        :param dataset: data from the current experience.
        :return:
        """
        print("Adatto la dimensione del classificatore ")
        in_features = self.classifier.in_features
        old_nclasses = self.classifier.out_features
        new_nclasses = max(self.classifier.out_features,
                           max(dataset.targets) + 1)

        if old_nclasses == new_nclasses:
            return
        old_w, old_b = self.classifier.weight, self.classifier.bias
        self.classifier = torch.nn.Linear(in_features, new_nclasses, self.bias)
        self.classifier.weight[:old_nclasses] = old_w

        if(self.bias):
            self.classifier.bias[:old_nclasses] = old_b


    def forward(self, x, **kwargs):
        """ compute the output given the input `x`. This module does not use
        the task label.

        :param x:
        :return:
        """
        if(self.norm_weights):
            self.classifier.weight.data = F.normalize(self.classifier.weight.data) 
        return self.classifier(x)

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
        
        x = x.view(-1, 128*3*3)
        x = self.prelu_fc1(self.fc1(x))
        y = self.inc_classifier(x)

        return y, x

def test():
    net = LeNet_PP(2)
    features, y = net(torch.randn(1, 1, 28, 28))
    print(y.size())
    print(features.size())


if __name__ == "__main__":
    test()