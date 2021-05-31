import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import requests
from torchvision import datasets, transforms


# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training=True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_set = datasets.MNIST('./data', train=True, download=True,
                               transform=custom_transform)
    test_set = datasets.MNIST('./data', train=False,
                              transform=custom_transform)

    if training:
        return torch.utils.data.DataLoader(train_set, batch_size=50)
    else:
        return torch.utils.data.DataLoader(test_set, batch_size=50, shuffle=False)


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """

    model = nn.Sequential(nn.Flatten(), nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64,
                                                                                                                   10))
    return model


def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(T):  # loop over the dataset multiple times
        correct = 0
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            # print statistics
            running_loss += loss.item()
            correct += (pred == labels).sum().item()

        print ("Train Epoch: %d Accuracy: %d/60000(%.2f%%) Loss: %.3f" % (epoch, correct, correct*100/60000, running_loss/60000))


def evaluate_model(model, test_loader, criterion, show_loss=True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            total += labels.size(0)
            running_loss += loss.item()
            correct += (predicted == labels).sum().item()

    if show_loss:
        print ("Average loss: %.4f" % (running_loss/total))

    print("Accuracy: %.2f%%" % (100 * correct/total))


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    list_probs = []
    logits = model(test_images)
    prob = F.softmax(logits, dim=1)

    for x in range (0, 10):
        list_probs.append((class_names[x], prob[index][x].item()*100))
    list_probs.sort(key=lambda x: x[1], reverse=True)

    for x in range (0, 3):
        print ( "%s: %.2f%%" % (list_probs[x][0], list_probs[x][1] ))


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    """model = build_model()
    train_loader = get_data_loader()
    print(type(train_loader))
    print(train_loader.dataset)
    test_loader = get_data_loader(False)
    print(test_loader.dataset)
    train_model(model, train_loader, criterion, T=5)
    evaluate_model(model, test_loader, criterion, show_loss=True)

    a = []
    for dat in test_loader:
        imgs, labels = dat
        a.append(imgs)

    my_tensor = torch.cat(a, dim=0)
    predict_label(model, my_tensor, 9999)"""
