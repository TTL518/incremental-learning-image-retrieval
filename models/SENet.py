'''SENet in PyTorch.

SENet is the winner of ImageNet-2017. The paper is not released yet.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging

logger = logging.getLogger('senet-forward')


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes // 16, kernel_size=1)  # Use nn.Conv2d instead of nn.Linear
        self.fc2 = nn.Conv2d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w  # New broadcasting feature from v0.2!
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )

        # SE layers
        self.fc1 = nn.Conv2d(planes, planes // 16, kernel_size=1)
        self.fc2 = nn.Conv2d(planes // 16, planes, kernel_size=1)

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))

        # Squeeze
        w = F.avg_pool2d(out, out.size(2))
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        # Excitation
        out = out * w

        out += shortcut
        return out


class SENet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, feat_dim=2048, current_classes=None, old_fc=None):
        super(SENet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512, num_classes)

        # TODO try to fix also these weights
        self.fc = nn.Linear(512, feat_dim, bias=False)
        
        # fc_classifier, logit
        self.classifier = nn.Linear(feat_dim, num_classes, bias=False)
        
        self.current_classes = current_classes

        if old_fc is not None:
            self.old_classifier = nn.Linear(feat_dim, num_classes, bias=False)
            self.old_classifier.weight = torch.nn.Parameter(torch.FloatTensor(old_fc.cpu()), requires_grad=True)
            # fix old classifier weights
            for para in self.old_classifier.parameters():
                para.requires_grad = False
        else:
            self.old_classifier = None

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x, ind=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        embeddings = self.fc(out)
        self.classifier.weight.data[self.current_classes] = F.normalize(
            self.classifier.weight.data[self.current_classes])
        # TODO qua è da normalizzare anche gli embedding
        # embeddings = F.normalize(embeddings, p=2, dim=1)
        out = self.classifier(embeddings)

        if ind is not None and len(ind) > 0 and self.old_classifier is not None:
            embeddings_BCT = embeddings[ind]
            outputs_BCT = self.old_classifier(embeddings_BCT)
        else:
            outputs_BCT = None
        return embeddings, out, outputs_BCT


def SENet18(num_classes=10, feat_dim=2048, current_classes=None, old_fc=None):
    return SENet(PreActBlock, [2, 2, 2, 2], num_classes, feat_dim, current_classes, old_fc)


def test():
    net = SENet18()
    e, y, _ = net(torch.randn(14, 3, 32, 32))
    print(e.size())
    print(y.size())
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(count_parameters(net))


if __name__ == "__main__":
    test()
    