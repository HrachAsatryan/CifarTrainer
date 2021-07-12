import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torch.optim as optim

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
    pass


class Model_3(nn.Module):
    pass


MODELS = {
    'model_1': Model_1,
    'model_2': Model_2,
    'model_3': Model_3
}


class CifarPytorchTrainer:
    """Implement training on CIFAR dataset"""

    DATASET_NAME = 'cifar'

    def __init__(self, model_name: str, epochs: int, lr: float, batch_size: int, train_on_gpu: bool, saving_directory :str):
        """
        Args:
            model_name: model_1, model_2 or model_3. Name of model we wish to implement
            epochs: number of epochs we wish to train for
            lr: the learning rate of our optimizer
        """
        self.model = MODELS[model_name]
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.train_on_gpu = train_on_gpu
    def train(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
        num_train = len(train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        train_sampler = SubsetRandomSampler(indices)

        train_loader = torch.utils.data.DataLoader(train_data, batch_size=self.batch_size, sampler=train_sampler)
        for epoch in range(1, self.epochs + 1):
            train_loss = 0.0
            self.model.train()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
            for data, target in train_loader:
                if self.train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)


    def infer(self, new_image: np.ndarray) -> np.ndarray:
        self.model.eval()
        output = self.model(new_image)
        return output

    def get_metrics(self) -> dict:
        # TODO: returns a metrics on train and validation data
        # f1 score, recall, precision, accuracy, balanced accuracy - all are in sickit learn
        pass

    def save(self):
        # TODO saves a model weights and metrics (as a JSON file)
        pass


if __name__ == "__main__":
    # TODO implement argparse
    pass

trainer = CifarPytorchTrainer('model_1', 5, 0.01, 32, False, 'asd')
CifarPytorchTrainer.train(self=trainer)
transform = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)
num_test = len(test_data)
indices = list(range(num_test))
np.random.shuffle(indices)
test_sampler = SubsetRandomSampler(indices)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=1, sampler=test_sampler)
dataiter = iter(test_loader)
image, label = dataiter.next()
image = image.numpy()
img = image.reshape(32, 32, 3)
print(img.shape)
trainer.infer(new_image=img)