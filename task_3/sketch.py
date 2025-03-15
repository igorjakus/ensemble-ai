import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from tqdm.notebook import tqdm
from task_dataset import load_dataset, TaskDataset  # This imports the TaskDataset class

# Make sure the TaskDataset class is registered in the main module's namespace
import sys
sys.modules['__main__'].TaskDataset = TaskDataset

dataset = load_dataset('task_3/Train.pt')
# visualization of the dataset is in dataset.ipynb
# (32, 32) is the size of the images
# 10 is the number of classes

# Define transforms
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert PIL image to tensor first
    # Use single-channel normalization if images are grayscale
    # transforms.Normalize((0.5,), (0.5,)) if dataset[0][0].mode == 'L' else transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Update dataset with transform
dataset.transform = transform

# Split dataset into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)

# Check number of classes
num_classes = len(set(dataset.labels))

# Create model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define model and replace final layer
model_name = "resnet18"  # Can also try resnet34 or resnet50
model = models.__dict__[model_name](pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def fgsm_attack(model, images, labels, epsilon=0.1):
    # Make sure gradients are calculated
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    loss = F.cross_entropy(outputs, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Create perturbation
    data_grad = images.grad.data
    sign_data_grad = data_grad.sign()
    
    # Create adversarial example
    perturbed_images = images + epsilon * sign_data_grad
    
    # Clamp to ensure valid pixel range [0,1]
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    
    return perturbed_images

def pgd_attack(model, images, labels, epsilon=0.1, alpha=0.01, num_iter=10):
    perturbed_images = images.clone().detach()
    
    for i in range(num_iter):
        perturbed_images.requires_grad = True
        
        # Forward pass
        outputs = model(perturbed_images)
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        if perturbed_images.grad is not None:
            perturbed_images.grad.data.zero_()
        loss.backward()
        
        # Create single-step perturbation
        data_grad = perturbed_images.grad.data
        adv_images = perturbed_images.detach() + alpha * data_grad.sign()
        
        # Project back to epsilon ball
        eta = torch.clamp(adv_images - images, -epsilon, epsilon)
        perturbed_images = torch.clamp(images + eta, 0, 1).detach()
    
    return perturbed_images

def train_epoch(model, dataloader, optimizer, criterion, device, adversarial=True, epsilon=0.03):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data in tqdm(dataloader):
        # Handle both 2-item and 3-item returns from dataloader
        if len(data) == 3:
            _, images, labels = data
        else:
            images, labels = data
        
        images, labels = images.to(device), labels.to(device)
        
        # Standard forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # If adversarial training is enabled
        if adversarial:
            # Generate adversarial examples with FGSM
            adv_images = fgsm_attack(model, images, labels, epsilon)
            adv_outputs = model(adv_images)
            adv_loss = criterion(adv_outputs, labels)
            
            # Generate adversarial examples with PGD
            pgd_images = pgd_attack(model, images, labels, epsilon, alpha=epsilon/5, num_iter=7)
            pgd_outputs = model(pgd_images)
            pgd_loss = criterion(pgd_outputs, labels)
            
            # Combined loss
            loss = loss + 0.5 * adv_loss + 0.5 * pgd_loss
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(dataloader)
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device, attack_type=None, epsilon=0.03):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for data in tqdm(dataloader):
        # Handle both 2-item and 3-item returns from dataloader
        if len(data) == 3:
            _, images, labels = data
        else:
            images, labels = data
        
        images, labels = images.to(device), labels.to(device)
        
        # If using attack
        if attack_type == "fgsm":
            # Need context manager for evaluation with gradients
            with torch.enable_grad():
                images.requires_grad = True
                images = fgsm_attack(model, images, labels, epsilon)
        elif attack_type == "pgd":
            # Need context manager for evaluation with gradients
            with torch.enable_grad():
                images = pgd_attack(model, images, labels, epsilon, alpha=epsilon/5, num_iter=10)
        
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    avg_loss = running_loss / len(dataloader)
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10, adversarial=True):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 
              'val_fgsm_loss': [], 'val_fgsm_acc': [], 'val_pgd_loss': [], 'val_pgd_acc': []}
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, adversarial)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        
        # Evaluate on clean validation data
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Evaluate on FGSM adversarial examples
        fgsm_loss, fgsm_acc = evaluate(model, val_loader, criterion, device, "fgsm")
        history['val_fgsm_loss'].append(fgsm_loss)
        history['val_fgsm_acc'].append(fgsm_acc)
        print(f"FGSM Loss: {fgsm_loss:.4f}, FGSM Acc: {fgsm_acc:.2f}%")
        
        # Evaluate on PGD adversarial examples
        pgd_loss, pgd_acc = evaluate(model, val_loader, criterion, device, "pgd")
        history['val_pgd_loss'].append(pgd_loss)
        history['val_pgd_acc'].append(pgd_acc)
        print(f"PGD Loss: {pgd_loss:.4f}, PGD Acc: {pgd_acc:.2f}%")
        
        # Save the best model based on average accuracy across clean and adversarial examples
        avg_acc = (val_acc + fgsm_acc + pgd_acc) / 3
        if avg_acc > best_acc:
            best_acc = avg_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_class': model_name
            }, 'best_model.pt')
            print(f"Saved new best model with avg acc: {avg_acc:.2f}%")
    
    return history

# Train model with adversarial training
history = train_model(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=10, adversarial=True)

def plot_history(history):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.plot(history['val_fgsm_loss'], label='FGSM Loss')
    plt.plot(history['val_pgd_loss'], label='PGD Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.plot(history['val_fgsm_acc'], label='FGSM Acc')
    plt.plot(history['val_pgd_acc'], label='PGD Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

plot_history(history)

# Load the best model
checkpoint = torch.load('best_model.pt')
model_class = checkpoint['model_class']
model = models.__dict__[model_class]()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(checkpoint['model_state_dict'])

# Final evaluation
print("Final Evaluation:")
model.to(device)
clean_loss, clean_acc = evaluate(model, val_loader, criterion, device)
fgsm_loss, fgsm_acc = evaluate(model, val_loader, criterion, device, "fgsm")
pgd_loss, pgd_acc = evaluate(model, val_loader, criterion, device, "pgd")

print(f"Clean Accuracy: {clean_acc:.2f}%")
print(f"FGSM Accuracy: {fgsm_acc:.2f}%")
print(f"PGD Accuracy: {pgd_acc:.2f}%")
print(f"Average Accuracy: {(clean_acc + fgsm_acc + pgd_acc) / 3:.2f}%")

# Save final submission
torch.save({
    'model_state_dict': model.state_dict(),
    'model_class': model_class
}, 'submission.pt')
print("Saved submission.pt file.")