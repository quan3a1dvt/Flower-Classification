
import comet_ml
import argparse
import numpy
import torch
from torch.nn import *
import torch.nn.functional as F
from dataset import *
from model import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from config import experiment
import cv2
from PIL import Image
import torchvision
import torchvision.transforms as transform
import os
import tqdm
from torchsummary import summary


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

def getImgData(X_batch, y_batch, dataset, args):
    X_batch_data = []
    for idx, img_name in enumerate(X_batch):
        path = os.path.join(args.data_dir, args.labels[y_batch[idx]])
        img = Image.open(os.path.join(path, img_name))
        resized_arr = preprocess[dataset](img)
        X_batch_data.append(resized_arr)
    X_batch_data = torch.stack(X_batch_data)
    #X_batch_data = torch.reshape(X_batch_data, (-1, 3, 224, 224))
    return X_batch_data

def save_model(model, path, name):
    torch.save(model.state_dict(), os.path.join(path, name))

def arugment_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=2e-6)
    parser.add_argument('--epoch', type=float, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    return parser

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(args):
    args.device = device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    args.labels = ['dandelion', 'daisy', 'tulip', 'sunflower', 'rose']
    args.data_dir = "data/flowers"
    X = []
    y = []
    for label in args.labels:
        path = os.path.join(args.data_dir, label)
        class_num = args.labels.index(label)
        for img in os.listdir(path):
            X.append(img)
            y.append(class_num)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    train_dataset = FlowerDataset(X_train, y_train)
    test_dataset = FlowerDataset(X_test, y_test)

    train_dl = DataLoader(train_dataset,
                          batch_size=args.batch_size,
                          num_workers=4,
                          shuffle=True,
                          collate_fn=FlowerDataset.pack)
    test_dl = DataLoader(test_dataset,
                         batch_size=args.batch_size,
                         num_workers=4,
                         shuffle=False,
                         collate_fn=FlowerDataset.pack)
    version = 'ResNet50-scratch'
    # model = torchvision.models.resnet50(weights=torchvision.models.resnet.ResNet50_Weights.DEFAULT)

    # for parameter in model.parameters():
    #     parameter.require_grad = True

    # classifier = nn.Sequential(
    #     nn.Linear(in_features=model.fc.in_features, out_features=5),
    #     nn.Sigmoid()
    # )
    # model.fc = classifier
    # model.load_state_dict(torch.load(f'checkpoints/{version}_best.pth'))
    model = ResNetModel(18, 3, 5)
    #model = CNNModel()
    #print(summary(model, (3, 224, 224)))
    model = model.to(device)
    ce = CrossEntropyLoss()
    
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=7, verbose=1)
    global_iter = 0
    best_test = {'p': 0.0, 'r': 0.0, 'f': 0.0}
    step = 0
    for epoch in range(args.epoch):
        model.train()
        print(get_lr(optimizer))
        labels = list(range(0, len(args.labels)))
        all_golds = []
        all_preds = []
        Loss = 0
        bar = tqdm.tqdm(train_dl, desc='Training', total=len(train_dl))
        for batch in bar:
            global_iter += 1
            Name_batch, y_batch = batch['img_name'], batch['label']
            X_batch = getImgData(Name_batch, y_batch, 'train', args)

            output = model(X_batch.to(device).float())
            all_golds += y_batch
            predicted = torch.max(output.data, 1)[1]
            all_preds += predicted.detach().cpu().numpy().tolist()
            y_batch = torch.Tensor(y_batch).type(torch.LongTensor).to(device)

            loss = ce(output, y_batch)
            Loss += loss.detach().cpu().numpy()

            if global_iter % 10 == 0:
                l = loss.detach().cpu().numpy()
                bar.set_description(f'Training: Loss={l:.4f}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print(all_golds)
        # print(all_preds)
        #scheduler.step(Loss / len(train_dl))
        perfs = metrics(all_golds, all_preds, labels)
        step = epoch
        experiment.log_metric('train' + '_' + 'precision', perfs['p'], step=step)
        experiment.log_metric('train' + '_' + 'recall', perfs['r'], step=step)
        experiment.log_metric('train' + '_' + 'f1', perfs['f'], step=step)

        experiment.log_metric("train_loss", Loss / len(train_dl), step=step)

        # Evaluation
        test_perf = evaluate(model, test_dl, args, 'test', global_iter, ce, step)
        step += 1

        if test_perf['f'] > best_test['f']:
            best_test = test_perf
            print('New best test @ {}'.format(epoch))
            save_model(model, 'checkpoints', f'{version}_best.pth')



def metrics(all_golds, all_preds, labels):
    p = precision_score(all_golds, all_preds, labels=labels, zero_division=0, average='micro')
    r = recall_score(all_golds, all_preds, labels=labels, zero_division=0, average='micro')
    f = f1_score(all_golds, all_preds, labels=labels, zero_division=0, average='micro')
    return {'p': p * 100, 'r': r * 100, 'f': f * 100}


def evaluate(model, dl, args, msg='test', global_iter=0, ce=None, step=0):
    model.eval()
    all_golds = []
    all_preds = []

    labels = list(range(0, len(args.labels)))
    device = args.device
    Loss = 0
    for batch in tqdm.tqdm(dl, desc=msg):
        Name_batch, y_batch = batch['img_name'], batch['label']
        X_batch = getImgData(Name_batch, y_batch, msg, args)

        output = model(X_batch.to(device).float())

        all_golds += y_batch
        predicted = torch.max(output.data, 1)[1]
        all_preds += predicted.detach().cpu().numpy().tolist()
        y_batch = torch.Tensor(y_batch).type(torch.LongTensor).to(device)
        loss = ce(output, y_batch)
        Loss += loss.detach().cpu().numpy()


    experiment.log_metric(msg.lower()+"_loss", Loss / len(dl), step=step)
    perfs = metrics(all_golds, all_preds, labels)
    experiment.log_metric(msg.lower() + '_' + 'precision', perfs['p'], step=step)
    experiment.log_metric(msg.lower() + '_' + 'recall', perfs['r'], step=step)
    experiment.log_metric(msg.lower() + '_' + 'f1', perfs['f'], step=step)
    print('{}: {:.2f} {:.2f} {:.2f} '.format(msg,
                                             perfs['p'],
                                             perfs['r'],
                                             perfs['f'],
                                             ))
    return perfs


if __name__ == '__main__':
    # sys.stdout = open("nohup.txt", "w")
    args = arugment_parser().parse_args()
    train(args)
    # sys.stdout.close()

