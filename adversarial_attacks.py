import torch
import torch.nn as nn

def FSGM_attack(batch_x, loss_f, eps):
    """
    Performs the Fast Sign Gradient Method (FSGM) attack on the input batch_x.

    Args:
        batch_x (torch.Tensor): The input batch of images.
        loss_f (callable): The loss function used to compute the loss.
        eps (float): The magnitude of the perturbation.

    Returns:
        torch.Tensor: The perturbed batch of images.

    """
    loss = loss_f(batch_x = batch_x)
    
    input_gradients = torch.autograd.grad(loss, batch_x)[0]
        
    input_gradient_sign = torch.sign(input_gradients)

    perturbed_batch_x = torch.clamp(batch_x + eps * input_gradient_sign, min=0, max=1).detach_()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define the device
    
    perturbed_batch_x = perturbed_batch_x.to(device)

    return perturbed_batch_x

# Adapted version of https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgd.py 
def PGDLInf_attack(batch_x, loss_f, iters, eps, alpha, random_start):
    """
    Performs the Projected Gradient Descent (PGD) attack with L-infinity norm on a batch of input images.

    Args:
        batch_x (torch.Tensor): The batch of input images.
        loss_f (callable): The loss function to maximize.
        iters (int): The number of iterations for the attack.
        eps (float): The maximum perturbation allowed for each pixel.
        alpha (float): The step size for each iteration of the attack.
        random_start (bool): Whether to start the attack from a random point.

    Returns:
        torch.Tensor: The perturbed batch of input images.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Define the device
    
    ori_images = batch_x.clone().detach().to(device).to(device)
    perturbed_batch_x = batch_x.clone().detach().to(device)
    
    if random_start:
        # Starting at a uniformly random point
        perturbed_batch_x = perturbed_batch_x + torch.empty_like(perturbed_batch_x).uniform_(
            -eps, eps
        )
        perturbed_batch_x = torch.clamp(perturbed_batch_x, min=0, max=1).detach()
    
    
    for _ in range(iters):
        perturbed_batch_x.requires_grad = True
        
        loss = loss_f(batch_x = perturbed_batch_x)
        
        input_gradients = torch.autograd.grad(loss, perturbed_batch_x)[0]
        input_gradient_sign = torch.sign(input_gradients)

        perturbed_batch_x = perturbed_batch_x.detach() + alpha * input_gradient_sign
        delta = torch.clamp(perturbed_batch_x - ori_images, min=-eps, max=eps)
        perturbed_batch_x = torch.clamp(ori_images + delta, min=0, max=1).detach()
        
    return perturbed_batch_x

#Adapted version of https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/pgdl2.py
def PGDL2_attack(batch_x, loss_f, iters, eps, alpha, random_start):
    """
    Performs the PGD with L2 norm adversarial attack on a batch of input images.

    Args:
        batch_x (torch.Tensor): The batch of input images.
        loss_f (callable): The loss function used to compute the loss.
        iters (int): The number of iterations for the attack.
        eps (float): The maximum perturbation allowed for each pixel.
        alpha (float): The step size for each iteration of the attack.
        random_start (bool): Whether to start the attack from a random point.

    Returns:
        torch.Tensor: The perturbed batch of input images.

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    
    eps_for_division=1e-10
    batch_size = len(batch_x)
    
    ori_images = batch_x.clone().detach().to(device)
    perturbed_batch_x = batch_x.clone().detach().to(device)
    
    if random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(perturbed_batch_x).normal_()
            d_flat = delta.view(perturbed_batch_x.size(0), -1)
            n = d_flat.norm(p=2, dim=1).view(perturbed_batch_x.size(0), 1, 1, 1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * eps
            perturbed_batch_x = torch.clamp(perturbed_batch_x + delta, min=0, max=1).detach()
    
    for _ in range(iters):
        
        perturbed_batch_x.requires_grad = True
        
        loss = loss_f(batch_x=perturbed_batch_x)
        
        # Update adversarial images
        grad = torch.autograd.grad(
            loss, perturbed_batch_x, retain_graph=False, create_graph=False
        )[0]
        grad_norms = (
            torch.norm(grad.view(batch_size, -1), p=2, dim=1)
            + eps_for_division
        )  # nopep8
        grad = grad / grad_norms.view(batch_size, 1, 1, 1)
        perturbed_batch_x = perturbed_batch_x.detach() + alpha * grad

        delta = perturbed_batch_x - ori_images
        delta_norms = torch.norm(delta.view(batch_size, -1), p=2, dim=1)
        factor = eps / delta_norms
        factor = torch.min(factor, torch.ones_like(delta_norms))
        delta = delta * factor.view(-1, 1, 1, 1)

        perturbed_batch_x = torch.clamp(ori_images + delta, min=0, max=1).detach()
            
    return perturbed_batch_x

def sinifgsm(model, images, labels, eps=8/255, alpha=2/255, steps=10, decay=1.0, m=5, targeted=False, device=None):
    """
    SI-NI-FGSM attack as described in 'NESTEROV ACCELERATED GRADIENT AND SCALE-INVARIANCE FOR ADVERSARIAL ATTACKS'.
    
    Args:
        model (torch.nn.Module): The target model.
        images (torch.Tensor): Input images of shape (N, C, H, W), values in range [0,1].
        labels (torch.Tensor): True labels for untargeted attack, target labels for targeted attack.
        eps (float): Maximum perturbation. Default: 8/255.
        alpha (float): Step size. Default: 2/255.
        steps (int): Number of iterations. Default: 10.
        decay (float): Momentum factor. Default: 1.0.
        m (int): Number of scale copies. Default: 5.
        targeted (bool): Whether to perform a targeted attack. Default: False.
        device (str or torch.device, optional): Device to run computations on.

    Returns:
        torch.Tensor: Adversarial examples of the same shape as `images`.
    """
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    momentum = torch.zeros_like(images).to(device)
    loss_fn = nn.CrossEntropyLoss()
    
    adv_images = images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        nes_image = adv_images + decay * alpha * momentum
        
        adv_grad = torch.zeros_like(images).to(device)

        for i in range(m):
            scaled_images = nes_image / (2 ** i)
            outputs = model(scaled_images)
            
            loss = -loss_fn(outputs, labels) if targeted else loss_fn(outputs, labels)
            
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
            adv_grad += grad
        
        adv_grad = adv_grad / m
        
        # Update adversarial images
        grad = decay * momentum + adv_grad / torch.mean(torch.abs(adv_grad), dim=(1,2,3), keepdim=True)
        momentum = grad
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images

def deepfool_attack(batch_x, loss_f, steps=50, overshoot=0.02):
    """
    Implements the DeepFool attack as described in 'DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks'.
    
    Args:
        batch_x (torch.Tensor): Input images of shape (N, C, H, W), values in range [0,1].
        loss_f (callable): The loss function used to compute the loss.
        steps (int): Maximum number of iterations. Default: 50.
        overshoot (float): Parameter for enhancing the noise. Default: 0.02.

    Returns:
        torch.Tensor: The perturbed batch of images.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_x = batch_x.clone().detach().to(device)
    adv_images = batch_x.clone().detach()
    
    for _ in range(steps):
        adv_images.requires_grad = True
        loss = loss_f(batch_x=adv_images)
        
        if adv_images.grad is not None:
            adv_images.grad.zero_()
        loss.backward(retain_graph=True)
        
        grad = adv_images.grad.clone().detach()
        
        # Calculate perturbation for the entire batch
        grad_norm = torch.norm(grad.view(grad.size(0), -1), p=2, dim=1)
        grad_norm = grad_norm.view(-1, 1, 1, 1)
        delta = grad / (grad_norm + 1e-8)
        
        # Update adversarial images
        adv_images = adv_images + (1 + overshoot) * delta
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    
    return adv_images

def EADEN_attack(batch_x, loss_f, kappa=0, lr=0.01, binary_search_steps=9, max_iterations=100, abort_early=True, initial_const=0.001, beta=0.001):
    """
    Implements the EAD (Elastic-Net) attack as described in 'EAD: Elastic-Net Attacks to Deep Neural Networks'.
    
    Args:
        batch_x (torch.Tensor): Input images of shape (N, C, H, W), values in range [0,1].
        loss_f (callable): The loss function used to compute the loss.
        kappa (float): Confidence parameter for the attack. Default: 0.
        lr (float): Learning rate for gradient descent. Default: 0.01.
        binary_search_steps (int): Number of binary search steps. Default: 9.
        max_iterations (int): Maximum number of iterations. Default: 100.
        abort_early (bool): Whether to abort early if not improving. Default: True.
        initial_const (float): Initial constant for binary search. Default: 0.001.
        beta (float): Trade-off parameter between L1 and L2. Default: 0.001.

    Returns:
        torch.Tensor: The perturbed batch of images.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_x = batch_x.clone().detach().to(device)
    batch_size = len(batch_x)
    
    # Initialize variables for binary search
    lower_bound = torch.zeros(batch_size, device=device)
    const = torch.ones(batch_size, device=device) * initial_const
    upper_bound = torch.ones(batch_size, device=device) * 1e10
    
    # Initialize adversarial images
    final_adv_images = batch_x.clone()
    best_l1 = torch.ones(batch_size, device=device) * 1e10
    best_score = torch.ones(batch_size, device=device) * -1
    
    # Initialization for FISTA
    x_k = batch_x.clone()
    y_k = batch_x.clone()
    
    for outer_step in range(binary_search_steps):
        prev_loss = 1e6
        current_lr = lr
        
        for iteration in range(max_iterations):
            y_k.requires_grad_(True)
            
            # Forward pass
            loss = loss_f(batch_x=y_k)
            
            # Add L2 regularization
            l2_loss = torch.norm((y_k - batch_x).view(batch_size, -1), p=2, dim=1).mean()
            total_loss = loss + const.mean() * l2_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient step
            with torch.no_grad():
                grad = y_k.grad.clone()
                y_k = y_k - current_lr * grad
                y_k = torch.clamp(y_k, min=0, max=1)
            
            # FISTA update
            t = (iteration + 1) / (iteration + 3)
            x_k_new = y_k
            y_k = x_k_new + t * (x_k_new - x_k)
            x_k = x_k_new
            
            # Update learning rate
            current_lr = lr * (1 - iteration / max_iterations) ** 0.5
            
            # Early stopping
            if abort_early and iteration % (max_iterations // 10) == 0:
                if total_loss > prev_loss * 0.999999:
                    break
                prev_loss = total_loss
            
            # Update best results
            with torch.no_grad():
                current_l1 = torch.norm((x_k - batch_x).view(batch_size, -1), p=1, dim=1)
                mask = (current_l1 < best_l1)
                best_l1[mask] = current_l1[mask]
                final_adv_images[mask] = x_k[mask]
        
        # Binary search update
        with torch.no_grad():
            mask = (best_score == -1)
            upper_bound[mask] = torch.min(upper_bound[mask], const[mask])
            lower_bound[~mask] = torch.max(lower_bound[~mask], const[~mask])
            
            mask = upper_bound < 1e9
            const[mask] = (lower_bound[mask] + upper_bound[mask]) / 2
            const[~mask] = const[~mask] * 10
    
    return final_adv_images

def CW_attack(batch_x, loss_f, c=1, kappa=0, steps=50, lr=0.01):
    """
    Implements the Carlini & Wagner (CW) attack as described in 'Towards Evaluating the Robustness of Neural Networks'.
    
    Args:
        batch_x (torch.Tensor): Input images of shape (N, C, H, W), values in range [0,1].
        loss_f (callable): The loss function used to compute the loss.
        c (float): Parameter for box-constraint. Default: 1.
        kappa (float): Confidence parameter. Default: 0.
        steps (int): Number of optimization steps. Default: 50.
        lr (float): Learning rate for Adam optimizer. Default: 0.01.

    Returns:
        torch.Tensor: The perturbed batch of images.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    batch_x = batch_x.clone().detach().to(device)
    batch_size = len(batch_x)
    
    # Initialize w in tanh space
    w = inverse_tanh_space(batch_x).detach()
    w.requires_grad = True
    
    best_adv_images = batch_x.clone().detach()
    best_L2 = 1e10 * torch.ones((batch_size)).to(device)
    prev_cost = 1e10
    
    optimizer = torch.optim.Adam([w], lr=lr)
    
    for step in range(steps):
        # Get adversarial images
        adv_images = tanh_space(w)
        
        # Calculate L2 loss
        current_L2 = torch.norm((adv_images - batch_x).view(batch_size, -1), p=2, dim=1)
        L2_loss = current_L2.sum()
        
        # Calculate classification loss
        loss = loss_f(batch_x=adv_images)
        
        # Total cost
        cost = L2_loss + c * loss
        
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # Update best adversarial images
        mask = (best_L2 > current_L2.detach())
        best_L2 = torch.where(mask, current_L2.detach(), best_L2)
        
        mask = mask.view([-1] + [1] * (len(batch_x.shape) - 1))
        best_adv_images = torch.where(mask, adv_images.detach(), best_adv_images)
        
        # Early stop when loss does not converge
        if step % max(steps // 10, 1) == 0:
            if cost.item() > prev_cost:
                return best_adv_images
            prev_cost = cost.item()
    
    return best_adv_images

def tanh_space(x):
    """Convert from tanh space to image space."""
    return 1/2 * (torch.tanh(x) + 1)

def inverse_tanh_space(x):
    """Convert from image space to tanh space."""
    return torch.atanh(torch.clamp(x * 2 - 1, min=-1, max=1))



