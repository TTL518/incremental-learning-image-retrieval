import torch
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