import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import time

from adversary import fgsm_attack, pgd_attack


class Resnet18(nn.Module):
    def __init__(self, num_classes=10, device=torch.device("cpu")):
        super(Resnet18, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model_name = "resnet18"
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        # move model to device (GPU)
        self.device = device
        self.model = self.model.to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
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
        epsilon = 0.1
        alpha = 0.01
        iters = 5
        a_clear, a_fgsm, a_pgd = 0.4, 0.3, 0.3  # zrównoważone wagi
        
        metrics = {
            'train_losses': [], 'clean_losses': [], 'fgsm_losses': [], 'pgd_losses': []
        }
        
        for epoch in range(1, epochs + 1):
            start_time = time.time()

            self.model.train()
            epoch_loss = 0.0
            clean_loss_sum = 0.0
            fgsm_loss_sum = 0.0
            pgd_loss_sum = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                idxs, imgs, labels = batch
                imgs, labels = imgs.to(self.device), labels.to(self.device)

                # Standard forward pass
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                clean_loss_sum += loss.item()
                
                # FGSM Attack
                adv_inputs = fgsm_attack(self, imgs, labels, epsilon)
                adv_outputs = self.model(adv_inputs)
                adv_loss = self.criterion(adv_outputs, labels)
                fgsm_loss_sum += adv_loss.item()
                
                # PGD Attack
                pgd_inputs = pgd_attack(self, imgs, labels, epsilon, alpha, iters)
                pgd_outputs = self.model(pgd_inputs)
                pgd_loss = self.criterion(pgd_outputs, labels)
                pgd_loss_sum += pgd_loss.item()
                
                # Total loss
                total_loss = a_clear * loss + a_fgsm * adv_loss + a_pgd * pgd_loss

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                epoch_loss += total_loss.item()

            # Calculate average losses
            avg_loss = epoch_loss / len(dataloader)
            avg_clean_loss = clean_loss_sum / len(dataloader)
            avg_fgsm_loss = fgsm_loss_sum / len(dataloader)
            avg_pgd_loss = pgd_loss_sum / len(dataloader)
            
            # Save metrics
            metrics['train_losses'].append(avg_loss)
            metrics['clean_losses'].append(avg_clean_loss)
            metrics['fgsm_losses'].append(avg_fgsm_loss)
            metrics['pgd_losses'].append(avg_pgd_loss)
            
            end_time = time.time()
            print(f"{epoch=} /{epochs} | {avg_loss=:.4f} | {avg_clean_loss=:.4f} | {avg_fgsm_loss=:.4f} | {avg_pgd_loss=:.4f} | time: {int(end_time-start_time)}s")

            # Save model and metrics to the file
            torch.save(self.model.state_dict(), f"trained/{self.model_name}_epoch_{epoch+1}.pt")
            torch.save(metrics, f"trained/{self.model_name}_metrics_epoch_{epoch+1}.pt")

        # Save final model and metrics
        torch.save(self.model.state_dict(), f"trained/{self.model_name}.pt")
        torch.save(metrics, f"trained/{self.model_name}_metrics.pt")
        print(f"Model saved to trained/{self.model_name}.pt")
        
        return metrics

    def try_lrs(self, dataloader: DataLoader):
        lre = torch.linspace(-4, 0, 200)
        lrs = 10**lre
        losses = []

        for lr in lrs:
            self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
            loss = self.train(dataloader, 1)
            losses.append(loss)

        metrics = {'lrs': lrs, 'losses': losses}
        torch.save(metrics, f"trained/{self.model_name}_lr_search.pt")
