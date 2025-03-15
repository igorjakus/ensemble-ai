import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from tqdm.notebook import tqdm

from task_dataset import load_dataset, TaskDataset  # noqa !!! don't remove this line
from adversary import fgsm_attack, pgd_attack


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

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
dev_loader = DataLoader(dev_dataset, batch_size=256, shuffle=False, num_workers=0)


# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # TODO: może warto większy
        self.model_name = "resnet18"
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model = self.model.to(device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)


def evaluate(model: Resnet18, dataloader: DataLoader):
    model.eval()  # make sure model is in evaluation mode
    
    # Przygotuj liczniki dla wszystkich trzech metryk
    correct_clean, correct_fgsm, correct_pgd = 0, 0, 0
    total = 0
    
    for batch in dataloader:
        idxs, imgs, labels = batch
        imgs, labels = imgs.to(device), labels.to(device)
        
        # Clean accuracy (bez ataku)
        with torch.no_grad():
            outputs_clean = model(imgs)
            _, predicted_clean = torch.max(outputs_clean.data, 1)
            correct_clean += (predicted_clean == labels).sum().item()
        
        # FGSM accuracy
        adv_inputs_fgsm = fgsm_attack(model, imgs, labels, 0.1)
        with torch.no_grad():
            outputs_fgsm = model(adv_inputs_fgsm)
            _, predicted_fgsm = torch.max(outputs_fgsm.data, 1)
            correct_fgsm += (predicted_fgsm == labels).sum().item()
        
        # PGD accuracy
        adv_inputs_pgd = pgd_attack(model, imgs, labels, 0.1, 0.01, 10)
        with torch.no_grad():
            outputs_pgd = model(adv_inputs_pgd)
            _, predicted_pgd = torch.max(outputs_pgd.data, 1)
            correct_pgd += (predicted_pgd == labels).sum().item()
        
        total += labels.size(0)
    
    # Oblicz i wyświetl wszystkie metryki
    accuracy_clean = 100 * correct_clean / total
    accuracy_fgsm = 100 * correct_fgsm / total
    accuracy_pgd = 100 * correct_pgd / total
    
    print(f"{accuracy_clean=:.2f}%")
    print(f"{accuracy_fgsm=:.2f}%")
    print(f"{accuracy_pgd=:.2f}%")
    
    return accuracy_clean, accuracy_fgsm, accuracy_pgd


def train(model: Resnet18, dataloader: DataLoader, num_epochs: int):
    # hyperparameters TODO: find best hyperparameters (+ learning rate)
    epsilon = 0.1
    alpha = 0.01
    iters = 10
    a_clear, a_fgsm, a_pgd = 0.5, 0.25, 0.25
    
    # make sure the model is in training mode
    model.train()

    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            idxs, imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)

            # Standard forward pass
            outputs = model(imgs)
            loss = model.criterion(outputs, labels)
            
            # FGSM Attack
            adv_inputs = fgsm_attack(model, imgs, labels, epsilon)
            adv_outputs = model(adv_inputs)
            adv_loss = model.criterion(adv_outputs, labels)
            
            # PGD Attack
            pgd_inputs = pgd_attack(model, imgs, labels, epsilon, alpha, iters)
            pgd_outputs = model(pgd_inputs)
            pgd_loss = model.criterion(pgd_outputs, labels)
            
            # Total loss
            total_loss = a_clear * loss + a_fgsm * adv_loss + a_pgd * pgd_loss
            
            model.optimizer.zero_grad()
            total_loss.backward()
            model.optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {total_loss.item():.4f}")

        torch.save(model.state_dict(), f"{model.model_name}_best.pth")
    
    print("Training finished!")


model = Resnet18()
model.load_state_dict(torch.load("resnet18_best.pth", map_location=torch.device('cuda')))
evaluate(model, test_loader)
train(model, dev_loader, 20)
evaluate(model, test_loader)
