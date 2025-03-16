import torch
from torch.utils.data import DataLoader

from task_dataset import load_dataset, TaskDataset  # noqa !!! don't remove this line
from resnet import Resnet18


# LOAD DATASET AND CREATE DATALOADERS 
dataset = load_dataset('Train.pt')
num_classes = len(set(dataset.labels))

train_size = int(0.89 * len(dataset))
test_size = int(0.10 * len(dataset))
dev_size = len(dataset) - train_size - test_size

train_dataset, test_dataset, dev_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size, dev_size])
print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}, Dev: {len(dev_dataset)}")

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=16, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=True, num_workers=16, pin_memory=True)
dev_loader = DataLoader(dev_dataset, batch_size=512, shuffle=True, num_workers=16, pin_memory=True)


# USE GPU IF AVAILABLE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# MEAT OF THE CODE
resnet = Resnet18(num_classes=num_classes, device=device)

# Load saved model if exists
try:
    state_dict = torch.load('trained/resnet18.pt')
    resnet.model.load_state_dict(state_dict)
    print("Loaded saved model.")
except Exception as e:
    print(e)
    print("Model not loaded.")

# Train model
# resnet.train(dev_loader, epochs=2)
resnet.evaluate(test_loader)
resnet.train(train_loader, epochs=60)
