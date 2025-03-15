import torch


def fgsm_attack(model, images, labels, epsilon=0.1) -> torch.Tensor:
    # Make sure gradients are calculated
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    loss = model.criterion(outputs, labels)
    
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

def pgd_attack(model, images, labels, epsilon=0.1, alpha=0.01, num_iter=10) -> torch.Tensor:
    perturbed_images = images.clone().detach()
    
    for i in range(num_iter):
        perturbed_images.requires_grad = True
        
        # Forward pass
        outputs = model(perturbed_images)
        loss = model.criterion(outputs, labels)
        
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
