import torch
from torch.utils.data import DataLoader

from task_dataset import load_dataset, TaskDataset  # noqa !!! don't remove this line
from resnet import Resnet18


# visualization of the dataset is in dataset.ipynb
dataset = load_dataset('Train.pt')
num_classes = len(set(dataset.labels))

train_size = int(0.8 * len(dataset))
test_size = int(0.1 * len(dataset))
dev_size = len(dataset) - train_size - test_size

# TODO: upewnić się, że jest faktycznie losowo podzielony
train_dataset, test_dataset, dev_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size, dev_size])
print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}, Dev: {len(dev_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=True, num_workers=16)
dev_loader = DataLoader(dev_dataset, batch_size=256, shuffle=True, num_workers=16)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


resnet = Resnet18(num_classes=num_classes, device=device)
resnet.load_state_dict(torch.load(f"{resnet.model_name}.pth", map_location=torch.device('cuda')))
resnet.evaluate(test_loader)
resnet.train(dev_loader, epochs=20)
resnet.evaluate(test_loader)
