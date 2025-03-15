from typing import Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform

    def __getitem__(self, index) -> Tuple[int, torch.Tensor, int]:
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label

    def __len__(self):
        return len(self.ids)


def shuffle(dataset: TaskDataset):
    indices = torch.randperm(len(dataset))
    dataset.ids = [dataset.ids[i] for i in indices]
    dataset.imgs = [dataset.imgs[i] for i in indices]
    dataset.labels = [dataset.labels[i] for i in indices]

def load_dataset(path: str):
    dataset = torch.load(path, weights_only=False)
    dataset.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return dataset
