
import numpy
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from sklearn.model_selection import train_test_split
import os

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

    torch_X_train = torch.tensor(X_train)
    torch_y_train = torch.tensor(y_train)
    torch_X_test = torch.tensor(X_test)
    torch_y_test = torch.tensor(y_test)

    train = torch.utils.data.TensorDataset(torch_X_train, torch_y_train)
    test = torch.utils.data.TensorDataset(torch_X_test, torch_y_test)
    print(len(train))
    train_loader = torch.utils.data.DataLoader(train, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False)