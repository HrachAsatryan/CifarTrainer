import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, balanced_accuracy_score
import json


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

    def __init__(self, model_name: str, epochs: int, lr: float, batch_size: int, train_on_gpu: bool, saving_directory=''):
        """
        Args:
            model_name: model_1, model_2 or model_3. Name of model we wish to implement
            epochs: number of epochs we wish to train for
            lr: the learning rate of our optimizer
        """
        self.model = MODELS[model_name]()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.train_on_gpu = train_on_gpu
        self.saving_directory = saving_directory
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.train_data = datasets.CIFAR10('data', train=True, download=True, transform=self.transform)
        self.test_data = datasets.CIFAR10('data', train=False, download=True, transform=self.transform)

    def train(self):
        num_train = len(self.train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        train_sampler = SubsetRandomSampler(indices)

        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, sampler=train_sampler)
        for epoch in range(1, self.epochs + 1):
            train_loss = 0.0
            self.model.train()
            self.criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(self.model.parameters(), lr=self.lr)
            for data, target in train_loader:
                if self.train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * data.size(0)


    def infer(self, new_image: np.ndarray) -> np.ndarray:
        self.model.eval()
        image = torch.from_numpy(new_image)
        output = self.model(image)
        return output


    def get_metrics(self) -> dict:
        metrics = {}
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=len(self.test_data))
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=len(self.train_data))
        for data, target in test_loader:
            if self.train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            test_preds = [torch.where(probas == torch.max(probas)) for probas in output]
            test_target = target
        for data, target in train_loader:
            if self.train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            train_preds = [torch.where(probas == torch.max(probas)) for probas in output]
            train_target = target

        test_pred = torch.zeros_like(test_target)
        for i in range(len(test_pred)):
            test_pred[i] = test_preds[i][0]

        train_pred = torch.zeros_like(train_target)
        for i in range(len(train_pred)):
            train_pred[i] = train_preds[i][0]

        metrics["train_accuracy"] = accuracy_score(train_target, train_pred)
        metrics["test_accuracy"] = accuracy_score(test_target, test_pred)
        metrics["train_balanced_accuracy"] = balanced_accuracy_score(train_target, train_pred)
        metrics["test_balanced_accuracy"] = balanced_accuracy_score(test_target, test_pred)
        metrics["train_precision"] = precision_score(train_target, train_pred, average='micro')
        metrics["test_precision"] = precision_score(test_target, test_pred, average='micro')
        metrics["train_recall"] = recall_score(train_target, train_pred, average='micro')
        metrics["test_recall"] = recall_score(test_target, test_pred, average='micro')
        metrics["train_f1score"] = f1_score(train_target, train_pred, average='micro')
        metrics["test_f1score"] = f1_score(test_target, test_pred, average='micro')
        return metrics

    def save(self):
        if self.saving_directory != '':
            dir = self.saving_directory + '/'
        else:
            dir = self.saving_directory
        state_dict = {}
        for key in self.model.state_dict():
            state_dict[key] = self.model.state_dict()[key].tolist()
        saver = {}
        saver["Metrics"] = self.get_metrics()
        saver["Weights"] = state_dict
        with open(dir + 'data.json', 'w') as f:
            json.dump(saver, f)


if __name__ == "__main__":
    # TODO implement argparse
    pass

trainer = CifarPytorchTrainer('model_1', 5, 0.01, 32, False)
#CifarPytorchTrainer.train(self=trainer)
transform = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
         ])
num_test = len(trainer.test_data)
indices = list(range(num_test))
np.random.shuffle(indices)
test_sampler = SubsetRandomSampler(indices)
test_loader = torch.utils.data.DataLoader(trainer.test_data, batch_size=1, sampler=test_sampler)
dataiter = iter(test_loader)
image, label = dataiter.next()
image = image.numpy()
trainer.infer(new_image=image)
#print(trainer.get_metrics())
trainer.save()