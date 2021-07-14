import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms
import torch.optim as optim
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, balanced_accuracy_score
import json
import argparse

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


MODELS = {
    'model_1': Model_1,
    'model_2': Model_2
}


class CifarPytorchTrainer:
    """Implement training on CIFAR dataset"""

    DATASET_NAME = 'cifar'

    def __init__(self, model_name: str, epochs: int, lr: float, batch_size: int, saving_directory=''):
        """
        Args:
            model_name: model_1, model_2 or model_3. Name of model we wish to implement
            epochs: number of epochs we wish to train for
            lr: the learning rate of our optimizer
            batch_size: the batch size of our model
            saving_directory: the directory we want to save our model weights and metrics
        """
        if model_name != 'model_3':
            self.model = MODELS[model_name]()
        else:
            self.model = models.vgg16(pretrained=True)
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.classifier[6] = nn.Linear(4096, 10)
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.train_on_gpu = torch.cuda.is_available()
        self.saving_directory = saving_directory
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.train_data = datasets.CIFAR10('data', train=True, download=True, transform=self.transform)
        self.test_data = datasets.CIFAR10('data', train=False, download=True, transform=self.transform)

    def train(self):
        """
        trains the model we picked with the parameters in __init__
        """
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
        """
        Does inference on a single image and returns the probabilites of each class
        :param new_image: a 32x32 numpy array, which is the image
        :return: a 10x1 numpy array, which is the probabities for each class
        """
        self.model.eval()
        image = torch.from_numpy(new_image)
        output = self.model(image)
        return output


    def get_metrics(self) -> dict:
        """
        gets the following metrics for our model (both test and train): accuracy, precision, recall, f1 score, balanced accuracy
        :return: the metrics mentioned above
        """
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
        """
        saves the metrics stated in get_metrics and the weights of the trained model
        """
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
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="the model number which we want")
    parser.add_argument("epochs", type=int, help="the number of epochs we want our model to train for")
    parser.add_argument("lr", type=float, help="the learning rate of our optimizer")
    parser.add_argument("batch_size", type=int, help="the batch size for our model")
    args = parser.parse_args()
    trainer = CifarPytorchTrainer(args.model, args.epochs, args.lr, args.batch_size)
    trainer.train()
    print("Trained! Saving the weights and metrics.")
    trainer.save()
