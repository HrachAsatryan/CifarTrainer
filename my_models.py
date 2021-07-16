import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models


class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.dense1 = nn.Linear(4 * 4 * 64, 512)
        self.dense2 = nn.Linear(512, 10)

        # max pooling layer
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view([-1, 4 * 4 * 64])
        x = F.relu(self.dense1(x))
        x = self.dense2(x)
        return x


class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 5, padding=2)
        self.conv4 = nn.Conv2d(256,512, 5, padding=2, stride=5)
        self.conv5 = nn.Conv2d(512, 512, 3, padding=1)

        # dense layers
        self.dense1 = nn.Linear(512, 512)
        self.dense2 = nn.Linear(512, 128)
        self.dense3 = nn.Linear(128, 10)

        # dropout layer
        self.dropout = nn.Dropout(0.5)

        # pooling layer
        self.pool = nn.MaxPool2d(4,4)

        #batchnorm layer
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.batchnorm4 = nn.BatchNorm2d(512)

        self.batchnorm1d = nn.BatchNorm1d(512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.relu(self.conv3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = F.relu(self.conv4(x))
        x = self.batchnorm4(x)
        x = F.relu(self.conv5(x))
        x = self.batchnorm4(x)
        x = self.dropout(x)
        x = self.pool(x)

        x = x.view([-1, 512])
        x = F.relu(x)
        x = self.batchnorm1d(x)
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = F.relu(self.dense3(x))

        return x

def Model_3():
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[6] = nn.Linear(4096, 10)
    return model

MODELS = {
    'model_1': Model_1,
    'model_2': Model_2,
    'model_3': Model_3
}
