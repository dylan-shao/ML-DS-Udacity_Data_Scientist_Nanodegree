# Imports here
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from collections import OrderedDict
import os
from PIL import Image
import numpy as np

data_dir = 'flowers'
config = {
    'drop': 0.5,
    'train_dir': data_dir + '/train',
    'valid_dir': data_dir + '/valid',
    'test_dir': data_dir + '/test'
}


# TODO:
# 1. redesign the class, with better __init__, and static methods and variables
# 2. make some variables dynamic, like hidden layers variables, so people could change it from input
# 3. create print helper with uniform format

class ImageClassifier:
    # TODO: refactor to not use *_dir in the init, should call them in other methods like "train" method
    def __init__(self, data_dir, save_dir, arch, learning_rate, epochs):
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.arch = arch
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.transform_data()
        self.set_model()
        self.set_mapping()


    def set_mapping(self):
        with open('cat_to_name.json', 'r') as f:
            self.cat_to_name = json.load(f)
        self.class_to_idx = self.image_datasets[0].class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def train(self):
        print('------ Start trainning model with {} epochs-----'.format(self.epochs))
        self.do_deep_learning(self.model, self.dataloaders[0], self.epochs, 40, self.criterion, self.optimizer)
        print('------ Trainning model finished!-----')

    def transform_data(self):
        print('------ Transforming data-----')

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
        print('------ Transforming data finished-----')

    def set_model(self):
        print('------ Setting the model using {} architecture-----'.format(self.arch))
        self.model = getattr(models, self.arch)(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False

        # TODO: make the size number dynamic as input
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
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)
        self.model.to('cuda')
        print('------ Setting the model finished-----')


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

        # make dir if save_dir not exist, TODO handle exception, if exception happens, delete the save_dir folder
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        save_path = os.getcwd() + '/' + self.save_dir

        # this is for the dir in the online command line
        save_path_to_checkpoint = '{}/checkpoint.pth'.format(save_path)
        print('------ save_dir is {}, save path is {} '.format(self.save_dir, save_path))
        print('------ Saving checkpoint in {}'.format(save_path_to_checkpoint))

        # TODO save, input_size, output_size, hidden_layers in checkpoints
        checkpoints = {
            'data_dir': self.data_dir,
            'save_dir': self.save_dir,
            'arch': self.arch,
            'learning_rate': self.learning_rate,
            'epochs':self.epochs,
            'state_dict': model.state_dict()
        }
        torch.save(checkpoints, save_path_to_checkpoint)

        print('------ Saving checkpoint finished ------')


    def process_image(self,image):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''

        # TODO: Process a PIL image for use in a PyTorch model
        im = Image.open(image)

        im_short = np.min(im.size)
        im_long = np.max(im.size)
        resize_short = 256
        resize_long = im_long*resize_short // im_short

        im = im.rotate(30).resize((resize_short,resize_long))

        width, height = im.size
        left = (width - 224)/2
        top = (height - 224)/2
        right = left + 224
        bottom = top + 224
        im = im.crop((left, top, right, bottom))
        np_image = np.array(im)


        np_image = (np_image - np.mean(np_image))/np.std(np_image)
        return np_image.transpose()

    def set_state_dict(self, state_dict):
        print('------ Loading checkpoint start ------')

        self.model.load_state_dict(state_dict)

        print('------ Loading checkpoint finished ------')


    def predict(self, image_path,  topk=10):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
        '''
        print('------ Predicting start ------')

        self.model.eval()
        # TODO: Implement the code to predict the class from an image file
        image = self.process_image(image_path)
        image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
        image = image.unsqueeze_(0)
        with torch.no_grad():
            outputs = self.model(image)

        ps = torch.exp(outputs)
        probs, classes =  ps.topk(topk)

        probs = np.array(probs)[0]
        classes = np.array(classes)[0]
        class_names = list(map(self.cat_to_name.get, list(map(self.idx_to_class.get, classes))))

        print('probabilities are: {}'.format(probs))
        print('classes are: {}'.format(classes))
        print('The flower with largest probs is: {}'.format(class_names[probs.argmax()]))

        print('------ Predicting finished ------')

        return probs, classes
