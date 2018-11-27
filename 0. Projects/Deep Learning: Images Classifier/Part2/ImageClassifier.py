# Imports here
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict

data_dir = 'flowers'
config = {
    'drop': 0.5,
    'train_dir': data_dir + '/train',
    'valid_dir': data_dir + '/valid',
    'test_dir': data_dir + '/test'
}



class ImageClassifier:
    def __init__(self, dir):
        self.dir = dir
        self.transform_data()
        self.set_model()


    def train(self):
        print('---- Start trainning model-----')
        self.do_deep_learning(self.model, self.dataloaders[0], 3, 40, self.criterion, self.optimizer)
        print('---- Trainning model finished!-----')

    def transform_data(self):
        print('---- Transforming data-----')

        self.data_transforms = [
            transforms.Compose([transforms.RandomRotation(30),
                               transforms.RandomResizedCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])]),
            transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])]),
            transforms.Compose([transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225])])

        ]

        # TODO: Load the datasets with ImageFolder
        self.image_datasets = [
            datasets.ImageFolder(config['train_dir'], transform=self.data_transforms[0]),
            datasets.ImageFolder(config['valid_dir'], transform=self.data_transforms[1]),
            datasets.ImageFolder(config['test_dir'], transform=self.data_transforms[2])
        ]

        # TODO: Using the image datasets and the trainforms, define the dataloaders
        self.dataloaders = [
            torch.utils.data.DataLoader(self.image_datasets[0], batch_size=64, shuffle=True),
            torch.utils.data.DataLoader(self.image_datasets[1], batch_size=32),
            torch.utils.data.DataLoader(self.image_datasets[2], batch_size=32)
        ]
        print('---- Transforming data finished-----')

    def set_model(self):
        print('---- Setting the model-----')
        self.model = models.vgg16(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, 4096)),
                                  ('relu1', nn.ReLU()),
                                  ('dropout1', nn.Dropout(config['drop'])),
                                  ('fc2', nn.Linear(4096, 1000)),
                                  ('relu2', nn.ReLU()),
                                  ('dropout2', nn.Dropout(config['drop'])),
                                  ('fc3', nn.Linear(1000, 102)),
                                  ('output', nn.LogSoftmax(dim=1))
                                  ]))

        self.model.classifier = classifier
        self.criterion = nn.NLLLoss()
        # Only train the classifier parameters, feature parameters are frozen
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=0.001)
        print('---- Setting the model finished-----')


    def validation(self, model, validationloader, criterion, device = 'cuda'):
        test_loss = 0
        accuracy = 0
        for images, labels in validationloader:
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()

        return test_loss, accuracy

    def do_deep_learning(self, model, trainloader, epochs, print_every, criterion, optimizer, device='cuda'):
        epochs = epochs
        print_every = print_every
        steps = 0
        running_loss = 0

        # change to cuda
        model.to(device)

        for e in range(epochs):
            model.train()
            for images, labels in trainloader:
                steps += 1

                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                output = model.forward(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    # Make sure network is in eval mode for inference
                    model.eval()

                    # Turn off gradients for validation, saves memory and computations
                    with torch.no_grad():
                        validation_loss, accuracy = self.validation(model, self.dataloaders[1], criterion)

                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(validation_loss/len(self.dataloaders[1])),
                          "Validation Accuracy: {:.3f}".format(accuracy/len(self.dataloaders[1])))

                    running_loss = 0

                    # Make sure training is back on
                    model.train()
