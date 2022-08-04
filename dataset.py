
import numpy as np
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch
class FlowerDataset(Dataset):

    def __init__(self, imgs_name, labels):
        super(FlowerDataset, self).__init__()
        self.imgs_name = imgs_name
        self.labels = labels

    def __len__(self):
        return len(self.imgs_name)

    def __getitem__(self, item):
        img_name = self.imgs_name[item]
        label = self.labels[item]
        return {
            'img_name': img_name,
            'label': label,
        }

    @staticmethod
    def pack(items):
        mini_pack = {
            k: TS_TYPE([x[k] for x in items])
            for k, TS_TYPE in TENSOR_TYPES.items()
        }
        return mini_pack


def keep(items):
    return items


def list_to_tensor(items):
    tensors = [torch.from_numpy(item.numpy()) for item in items]
    tensors = torch.stack(tensors, dim=0)
    return tensors


def flatten(items):
    all_items = [y for x in items for y in x]
    return torch.LongTensor(all_items)


TENSOR_TYPES = {
    'img_name': keep,
    'label': keep,
}

if __name__ == '__main__':
    pass