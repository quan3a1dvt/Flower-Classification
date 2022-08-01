
import comet_ml
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from model import *
from sklearn.model_selection import train_test_split
from config import experiment
import cv2
from PIL import Image
import torchvision
import torchvision.transforms as transform
import os
import tqdm



def getImgData(X_batch, y_batch, data_dir, dat):
    preprocess = {
        'train': transform.Compose([
            transform.Resize([224, 224]),
            transform.ToTensor(),
            transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transform.Compose([
            transform.Resize([224, 224]),
            transform.ToTensor(),
            transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    X_batch_data = []
    for i in range(len(X_batch)):
        path = os.path.join(data_dir, labels[y_batch[i]])
        img = Image.open(os.path.join(path, img_name[X_batch[i]]))
        resized_arr = preprocess[dat](img)
        X_batch_data.append(resized_arr)
    X_batch_data = torch.stack(X_batch_data)
    #X_batch_data = torch.reshape(X_batch_data, (-1, 3, 224, 224))
    return X_batch_data


def fit(model, train_loader, dev_loader):
        EPOCHS = 200
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001, weight_decay = 1e-5)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, verbose=1)
        error = nn.CrossEntropyLoss()
        step = 0
        for epoch in range(EPOCHS):
            correct = 0
            Loss = 0
            count = 0
            bar1 = tqdm.tqdm(train_loader, desc='Training', total=len(train_loader))
            model.train()
            for batch_idx, (X_batch, y_batch) in enumerate(bar1):
                X_batch = getImgData(X_batch, y_batch, data_dir, 'train')

                var_X_batch = X_batch.to(device)
                var_y_batch = y_batch.to(device)
                optimizer.zero_grad()

                output = model(var_X_batch.float())
                loss = error(output, var_y_batch)

                loss.backward()
                optimizer.step()


                predicted = torch.max(output.data, 1)[1]
                correct += (predicted == var_y_batch).sum()

                count += len(predicted)
                Loss += loss.item()

            Loss /= len(train_loader)
            scheduler.step(loss)
            experiment.log_metric("train_loss", Loss, step=step)
            experiment.log_metric("correct_train", correct * 100 / count, step=step)
            Loss = 0
            correct = 0
            count = 0
            model.eval()
            bar2 = tqdm.tqdm(dev_loader, desc='Test', total=len(dev_loader))
            for batch_idx, (X_batch, y_batch) in enumerate(bar2):
                X_batch = getImgData(X_batch, y_batch, data_dir, 'test')

                var_X_batch = X_batch.to(device)
                var_y_batch = y_batch.to(device)

                optimizer.zero_grad()
                output = model(var_X_batch.float())
                loss = error(output, var_y_batch)
                Loss += loss.item()
                predicted = torch.max(output.data, 1)[1]
                correct += (predicted == var_y_batch).sum()
                count += len(predicted)

            Loss /= len(dev_loader)
            experiment.log_metric("test_loss", Loss, step=step)
            experiment.log_metric("correct_test", correct * 100 / count, step=step)
            step += 1


if __name__ == '__main__':
    labels = ['dandelion', 'daisy', 'tulip', 'sunflower', 'rose']
    data_dir = "data/flowers"
    X = []
    y = []
    img_name = []
    k = 0
    for label in labels:
        dat = []
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            if k == 1:
                print(img)
            img_name.append(img)
            X.append(k)
            y.append(class_num)
            k += 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    BATCH_SIZE = 32

    torch_X_train = torch.from_numpy(numpy.array(X_train))
    torch_y_train = torch.from_numpy(numpy.array(y_train))
    torch_X_test = torch.from_numpy(numpy.array(X_test))
    torch_y_test = torch.from_numpy(numpy.array(y_test))

    train = torch.utils.data.TensorDataset(torch_X_train, torch_y_train)
    test = torch.utils.data.TensorDataset(torch_X_test, torch_y_test)
    print(len(train))
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())

    model = torchvision.models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.require_grad = False

    classifier = nn.Sequential(
        nn.Linear(in_features=model.fc.in_features, out_features=5),
        nn.Sigmoid()
    )
    model.fc = classifier
    #model = CNNModel()
    model.to(device)

    fit(model, train_loader, test_loader)