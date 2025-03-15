import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models

from adversary import fgsm_attack, pgd_attack


class Resnet18(nn.Module):
    def __init__(self, num_classes=10, device=torch.device("cpu")):
        super(Resnet18, self).__init__()
        # TODO: może warto zwiększyć do Resnet50
        # TODO: może bez wag domyślnych??
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model_name = "resnet18"
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # move model to device (GPU)
        self.device = device
        self.model = self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def evaluate(self, dataloader: DataLoader):
        self.model.eval()  # make sure model is in evaluation mode
        
        correct_clean, correct_fgsm, correct_pgd = 0, 0, 0
        total = 0
        
        for batch in dataloader:
            idxs, imgs, labels = batch
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            
            # Clean accuracy (bez ataku)
            with torch.no_grad():
                outputs_clean = self.model(imgs)
                _, predicted_clean = torch.max(outputs_clean.data, 1)
                correct_clean += (predicted_clean == labels).sum().item()
            
            # FGSM accuracy
            adv_inputs_fgsm = fgsm_attack(self, imgs, labels, 0.1)
            with torch.no_grad():
                outputs_fgsm = self.model(adv_inputs_fgsm)
                _, predicted_fgsm = torch.max(outputs_fgsm.data, 1)
                correct_fgsm += (predicted_fgsm == labels).sum().item()
            
            # PGD accuracy
            adv_inputs_pgd = pgd_attack(self, imgs, labels, 0.1, 0.01, 10)
            with torch.no_grad():
                outputs_pgd = self.model(adv_inputs_pgd)
                _, predicted_pgd = torch.max(outputs_pgd.data, 1)
                correct_pgd += (predicted_pgd == labels).sum().item()
            
            total += labels.size(0)
        
        # Calculate accuracies
        accuracy_clean = 100 * correct_clean / total
        accuracy_fgsm = 100 * correct_fgsm / total
        accuracy_pgd = 100 * correct_pgd / total
        
        print(f"{accuracy_clean=:.2f}%")
        print(f"{accuracy_fgsm=:.2f}%")
        print(f"{accuracy_pgd=:.2f}%")
        
        return accuracy_clean, accuracy_fgsm, accuracy_pgd


    def train(self, dataloader: DataLoader, epochs: int):
        # hyperparameters 
        # TODO: find best hyperparameters (+ learning rate)
        epsilon = 0.1
        alpha = 0.01
        iters = 10
        a_clear, a_fgsm, a_pgd = 0.6, 0.25, 0.25
        
        # make sure the model is in training mode
        self.model.train()
        
        best_loss = float('inf')

        for epoch in range(epochs):
            for batch_idx, batch in enumerate(dataloader):
                idxs, imgs, labels = batch
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                # Standard forward pass
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                
                # FGSM Attack
                adv_inputs = fgsm_attack(self, imgs, labels, epsilon)
                adv_outputs = self.model(adv_inputs)
                adv_loss = self.criterion(adv_outputs, labels)
                
                # PGD Attack
                pgd_inputs = pgd_attack(self, imgs, labels, epsilon, alpha, iters)
                pgd_outputs = self.model(pgd_inputs)
                pgd_loss = self.criterion(pgd_outputs, labels)
                
                # Total loss
                total_loss = a_clear * loss + a_fgsm * adv_loss + a_pgd * pgd_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}")

            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(self.model.state_dict(), f"{self.model_name}_best.pth")
        
        torch.save(self.model.state_dict(), f"{self.model_name}.pth")
        print("Training finished!")
