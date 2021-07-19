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
import my_models

OPTIMIZERS = {
    'Adam': optim.Adam,
    'SGD': optim.SGD
}


class CifarPytorchTrainer:
    """Implement training on CIFAR dataset"""

    DATASET_NAME = 'cifar'

    def __init__(self, model_name: str, epochs: int, lr: float, batch_size: int, patience: int = 0, optimizer: str = "Adam", saving_directory: str = ''):
        """
        Args:
            model_name: model_1, model_2 or model_3. Name of model we wish to implement
            epochs: number of epochs we wish to train for
            lr: the learning rate of our optimizer
            batch_size: the batch size of our model
            optimizer of our model, default is Adam
            saving_directory: the directory we want to save our model weights and metrics, default is the same dir
        """
        self.model = my_models.MODELS[model_name]()
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
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = OPTIMIZERS[optimizer](self.model.parameters(), lr=self.lr)
        self.patience = patience

    def train(self):
        """
        trains the model we picked with the parameters in __init__
        """
        num_train = len(self.train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        train_sampler = SubsetRandomSampler(indices)

        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, sampler=train_sampler)
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size)
        train_losses = []
        test_losses = []
        for epoch in range(1, self.epochs + 1):
            train_loss = 0.0
            test_loss = 0.0
            self.model.train()
            for data, target in train_loader:
                if self.train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * data.size(0)
            print(f"Training loss for epoch {epoch} is {train_loss / len(train_loader.sampler)}.")
            for data, target in test_loader:
                if self.train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                output = self.model(data)
                loss = self.criterion(output, target)
                test_loss += loss.item() * data.size(0)
            print(f"Test loss for epoch {epoch} is {test_loss / len(test_loader.sampler)}.")
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            if test_loss <= np.min(test_losses):
                torch.save(self.model, 'model.pt')
            if abs(np.where(np.array(test_losses) == np.min(test_losses))[0] - epoch) > self.patience:
                self.model = torch.load('model.pt')
                break

    def infer(self, new_image: np.ndarray) -> np.ndarray:
        """
        Does inference on a single image and returns the probabilites of each class
        :param new_image: a 32x32 numpy array, which is the image
        :return: a 10x1 numpy array, which is the probabilities for each class
        """
        self.model.eval()
        new_image = new_image.reshape(32, 32, 3)
        image = self.transform(new_image)
        image = image.reshape(1, 3, 32, 32)
        output = self.model(image)
        return output

    def get_metrics(self) -> dict:
        """
        gets the following metrics for our model (both test and train): accuracy, precision, recall, f1 score, balanced accuracy
        :return: the metrics mentioned above
        """
        metrics = {}
        test_preds = []
        test_target = []
        train_preds = []
        train_target = []
        test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.batch_size)
        train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size)
        for data, target in test_loader:
            if self.train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            test_preds += [torch.where(probas == torch.max(probas)) for probas in output]
            test_target += target
        for data, target in train_loader:
            if self.train_on_gpu:
                data, target = data.cuda(), target.cuda()
            output = self.model(data)
            train_preds += [torch.where(probas == torch.max(probas)) for probas in output]
            train_target += target

        test_pred = torch.zeros_like(torch.Tensor(test_target))
        for i in range(len(test_pred)):
            test_pred[i] = test_preds[i][0]

        train_pred = torch.zeros_like(torch.Tensor(train_target))
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
    parser.add_argument("patience", type=int, help="the patience of our optimizer")
    parser.add_argument("optimizer", type=str, help="the optimizer of our model")
    parser.add_argument("saving_dir", type=str, help="the directory in which we wish to save our model weights and metrics")
    args = parser.parse_args()
    trainer = CifarPytorchTrainer(args.model, args.epochs, args.lr, args.batch_size, args.patience, args.optimizer, args.saving_dir)
    trainer.train()
    print("Trained! Saving the weights and metrics.")
    trainer.save()
