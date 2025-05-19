import torch
import numpy as np
import torch.nn.functional as F
from autoencoder_helpers import *

def prototype_loss(h_x, prototype_labels, y, lambda_param=0.5):
    """
    Computes L_h(x,y) = D(x,y) - λR(x,y)
    
    Args:
        h_x: distances from input to each prototype [batch_size x n_prototypes]
        prototype_labels: class labels for each prototype [n_prototypes]
        y: target class label [batch_size]
        lambda_param: weighting parameter for R term
    """
    # Convert to tensors if needed
    h_x = torch.as_tensor(h_x)
    prototype_labels = torch.as_tensor(prototype_labels)
    y = torch.as_tensor(y)
    
    # Move all tensors to the same device
    device = h_x.device
    prototype_labels = prototype_labels.to(device)
    y = y.to(device)
    
    # Ensure h_x is 2D [batch_size x n_prototypes]
    if h_x.dim() == 1:
        h_x = h_x.unsqueeze(0)
    
    # Ensure y is 1D [batch_size]
    if y.dim() == 0:
        y = y.unsqueeze(0)
    elif y.dim() > 1:
        y = y.squeeze()
        
    # Get batch size and number of prototypes
    batch_size = h_x.shape[0]
    n_prototypes = h_x.shape[1]
    
    # Ensure prototype_labels is 1D [n_prototypes]
    if prototype_labels.dim() > 1:
        prototype_labels = prototype_labels.squeeze()
    
    # Create the target mask with proper broadcasting
    y_expanded = y.view(batch_size, 1)  # [batch_size x 1]
    prototype_labels_expanded = prototype_labels.view(1, -1)  # [1 x n_prototypes]
    target_mask = (prototype_labels_expanded == y_expanded)  # [batch_size x n_prototypes]
    
    # Compute the loss terms
    squared_distances = h_x**2  # [batch_size x n_prototypes]
    
    # For each example in batch, compute D and R terms
    D_num = (squared_distances * target_mask).sum(dim=1)  # [batch_size]
    D_den = target_mask.sum(dim=1).clamp(min=1)  # [batch_size]
    D = torch.sqrt(D_num / D_den)  # [batch_size]
    
    nontarget_mask = ~target_mask  # [batch_size x n_prototypes]
    R_num = (squared_distances * nontarget_mask).sum(dim=1)  # [batch_size]
    R_den = nontarget_mask.sum(dim=1).clamp(min=1)  # [batch_size]
    R = torch.sqrt(R_num / R_den)  # [batch_size]
    
    # Compute final loss
    loss = D - lambda_param * R  # [batch_size]
    return loss.mean()


def get_ms(prototype_classes, batch_y, change_expl=None):
    examples_classes = batch_y.cpu().detach().numpy()
    
    # If force_class is integer, then we try to get closer to the prototype representing force_class
    if isinstance(change_expl, int):
        ms = torch.zeros((batch_y.shape[0], prototype_classes.shape[0]), dtype=torch.float)
        
        for e in range(examples_classes.shape[0]):
            for p in range(prototype_classes.shape[0]):
                if prototype_classes[p, change_expl] == 1:
                    ms[e, p] = 1
        return ms
    
    # If force_class is string, then we try to get closer to the prototype number force_class (P 'FC')
    if isinstance(change_expl, str):  
        ms = torch.zeros((batch_y.shape[0], prototype_classes.shape[0]), dtype=torch.float)      
        ms[:, int(change_expl[1:])-1] = 1
        return ms
    
    # If nothing, then we try to get away from the prototypes of the class of the example
    ms = torch.zeros((batch_y.shape[0], prototype_classes.shape[0]), dtype=torch.float)
    for e in range(examples_classes.shape[0]):
        for p in range(prototype_classes.shape[0]):
            if prototype_classes[p, examples_classes[e]] == 1:
                ms[e, p] = 1 #0
                
    return ms

# ADVL loss function
def Loss_1(model, batch_x, batch_y, alpha1, alpha2, force_class, change_expl, objective):

    """
    if objective == "cenc": change expl no change class
    if objective == cecc: change expl and change class
    if objective == necl: no change explanation change class
    if objective == "advl": adversarial explanation objective
    """
    
    if objective == "necc" and change_expl is not None:
        raise ValueError("If objective is necc, change_expl must be None")
    if objective == "cenc" and force_class is not None:
        raise ValueError("If objective is cenc, force_class must be None")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device
    
    pred_y = model.forward(batch_x)
    
    loss_function = torch.nn.CrossEntropyLoss() # Cross-entropy loss
    
    # Boolean matrix, where each row is a prototype and if a column i has a 1, means that the prototype is from class i
    thresh = -1.8 # We want to find prototypes for all the classes
    prototype_classes = np.where((model.fc.linear.weight.cpu().detach().numpy().T <= np.min(model.fc.linear.weight.cpu().detach().numpy().T, axis=1, keepdims=True)) | (model.fc.linear.weight.cpu().detach().numpy().T < thresh), 1, 0)
    prototype_labels = np.where(prototype_classes == 1)[1] # Labels of the prototypes

    feature_vectors = model.feature_vectors # Latent representation of the examples in the batch_x
    prototype_distances = model.prototype_layer.prototype_distances # Latent representation of the prototypes
    
    loss = 0
      
    ms = get_ms(prototype_classes, batch_y, change_expl=change_expl).to(device) # Importance matrix
    
    dists_loss = torch.sum(torch.matmul(ms, list_of_distances(prototype_distances, feature_vectors.view(-1, model.in_channels_prototype)))) # Weighted prototype distance loss

    # CHANGE explanation BUT KEEP prediction the same
    if objective == "cenc": 
        loss += -1 * alpha1 * loss_function(pred_y, batch_y)
        # penalize being too close to original prototypes (encourage explanation change)
        if change_expl is None:
            loss += alpha2 * dists_loss
        # reward if explanation already changed    
        elif change_expl is not None:
            loss += -1 * alpha2 * dists_loss

    # CHANGE explanation AND CHANGE prediction
    elif objective == "cecc":
        if force_class is None:
            loss += alpha1 * loss_function(pred_y, batch_y)
            
        elif force_class is not None:
            y_target = torch.full_like(batch_y, force_class)
            loss += -1 * alpha1 * loss_function(pred_y, y_target)	
        
        # penalize being too close to original prototypes (encourage explanation change)
        if change_expl is None:
            loss += alpha2 * dists_loss
            
        # reward if explanation already changed
        elif change_expl is not None:
            loss += -1 * alpha2 * dists_loss
      
    # KEEP explanation AND CHANGE prediction
    elif objective == "necc":
        
        if force_class is None:
            loss += alpha1 * loss_function(pred_y, batch_y)
            
        elif force_class is not None:
            y_target = torch.full_like(batch_y, force_class)
            loss += -1 * alpha1 * loss_function(pred_y, y_target)	
        
        loss += -1 * alpha2 * dists_loss
        
    # Adversarial explanation objective: max L_h(x*, y) - ξL_y(f(x*), y) - α||h(x*)||₂
    elif objective == "advl":
        # Get model predictions using the model's existing structure
        # First get the encoder output
        encoded = model.encoder(batch_x)
        
        # Reshape encoded to match prototype dimensions
        # Assuming encoded is [batch_size x hidden_dim]
        # and prototype_layer expects [batch_size x in_channels_prototype]
        encoded = encoded.view(encoded.shape[0], model.in_channels_prototype)
        
        # Then get prototype distances
        prototype_distances = model.prototype_layer(encoded)
        
        # Get predictions
        pred_y = model.forward(batch_x)
        
        if pred_y.dim() > 1:
            pred_y = torch.argmax(pred_y, dim=1)
        
        # Get number of prototypes and classes
        n_prototypes = prototype_distances.shape[1]
        n_classes = model.fc.linear.weight.shape[0]
        protos_per_class = n_prototypes // n_classes
        
        # Create prototype labels
        prototype_labels = torch.arange(n_classes, device=device).repeat_interleave(protos_per_class)
        
        # 1. Explanation loss L_h(x*, y)
        explanation_loss = prototype_loss(
            prototype_distances,  # [batch_size x n_prototypes]
            prototype_labels,     # [n_prototypes]
            pred_y               # [batch_size]
        )
        
        # 2. Classification loss
        classification_loss = loss_function(model.forward(batch_x), batch_y)
        
        # 3. Stability term
        stability_term = torch.norm(prototype_distances.view(batch_x.shape[0], -1), dim=1).mean()
        
        # Combine terms
        loss = explanation_loss - alpha1 * classification_loss - alpha2 * stability_term
        
    return loss

def CELoss(model, batch_x, batch_y):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_function = torch.nn.CrossEntropyLoss() # Cross-entropy loss
    pred_y = model.forward(batch_x)
    pred_y = pred_y.to(device)
    return loss_function(pred_y, batch_y)

"""
    each row in d corresponds to one prototype and each column to one example. we want to maximize the distance between prototypes to
    the examples that are not from the same class and minimize the distance between prototypes to the examples that are from the 
    same class. 
    
    we create 2 masks, the clst mask, for each example, if the example is from class i, then the positions corresponding to the
    distances between the prototypes of class i and the example are set to 1 and the rest to infinite. 
    the sep mask is just the opposite.
    
    this way, after applying the mask the distances in wich we are interested in each case will have the corresponding value and the
    rest infinite. beacuse we search for the minimun, the infnites are never going to be taken into account.
    
    """
    
def Clst_Sep_E1(model, batch_x, batch_y):
    MAX = 1e5
    feature_vectors = model.feature_vectors 
    prototype_distances =  model.prototype_layer.prototype_distances 
    n_prototypes = model.fc.linear.weight.size(1)
    n_prototypes_by_class = n_prototypes // 10
    
    d = list_of_distances(prototype_distances, feature_vectors.view(-1, model.in_channels_prototype))
    
    mask_clst = torch.full_like(d, MAX)
    mask_sep = torch.full_like(d, 1.0)
    
    for i in range(batch_x.shape[0]):
        ini = batch_y[i] * n_prototypes_by_class
        mask_clst[ini:ini + n_prototypes_by_class, i] = 1.0
        mask_sep[ini:ini + n_prototypes_by_class, i] = MAX
    
    dists_to_class_prototypes = torch.mul(d, mask_clst)
    dists_to_no_class_prototypes = torch.mul(d, mask_sep)
    
    clst = torch.mean(torch.min(dists_to_class_prototypes, dim=0)[0])  # minimum for each column
    sep = -torch.mean(torch.min(dists_to_no_class_prototypes, dim=0)[0])  # minimum for each column
    e1 = torch.mean(torch.min(dists_to_class_prototypes, dim=1)[0])
    return clst, sep, e1


#Adapted deafult loss replaicing train_e2 with clst and sep loss, for the balanced model
def ClstSepLoss(model, batch_x, batch_y, pred_y, lambda_class, lambda_1, lambda_clus, lambda_sep, lambda_ae, batch_x_ori=None):
    
    loss_function = torch.nn.CrossEntropyLoss()
    train_ce = loss_function(pred_y, batch_y)

    feature_vectors = model.feature_vectors

    clst_l, sep_l, train_e1 = Clst_Sep_E1(model, batch_x, batch_y)
    
    out_decoder = model.decoder(feature_vectors)
    
    if batch_x_ori is None:
        train_ae = torch.mean(list_of_norms(out_decoder-batch_x))
    else:
        train_ae = torch.mean(list_of_norms(out_decoder-batch_x_ori))

    train_te = lambda_class * train_ce +\
            lambda_1 * train_e1 +\
            lambda_clus * clst_l +\
            lambda_sep * sep_l +\
            lambda_ae * train_ae
            
    return train_te, train_ce, train_e1, clst_l, sep_l, train_ae


#Loss used by default to train PrototypeDL
def generalLoss(model, batch_x, batch_y, pred_y, lambda_class, lambda_1, lambda_2, lambda_ae, batch_x_ori=None):
    
    loss_function = torch.nn.CrossEntropyLoss()
    train_ce = loss_function(pred_y, batch_y)

    prototype_distances = model.prototype_layer.prototype_distances
    feature_vectors = model.feature_vectors

    if lambda_1 == 0:
        train_e1 = 0
    else:
        train_e1 = torch.mean(torch.min(list_of_distances(prototype_distances, feature_vectors.view(-1, model.in_channels_prototype)), dim=1)[0])
    
    train_e2 = torch.mean(torch.min(list_of_distances(feature_vectors.view(-1, model.in_channels_prototype ), prototype_distances), dim=1)[0])
    
    out_decoder = model.decoder(feature_vectors)
    
    if batch_x_ori is None:
        train_ae = torch.mean(list_of_norms(out_decoder-batch_x))
    else:
        train_ae = torch.mean(list_of_norms(out_decoder-batch_x_ori))
    
    train_ae = torch.mean(list_of_norms(out_decoder-batch_x))

    train_te = lambda_class * train_ce +\
            lambda_1 * train_e1 +\
            lambda_2 * train_e2 +\
            lambda_ae * train_ae
    
    return train_te, train_ce, train_e1, train_e2, train_ae


# Loss for the attack to the interpretability of the model 
# - - - - - - - - - - - - - - -  - - - - - 
def compute_prototype_distances(x, prototype_layer):
    """
    Compute distances between input x and all prototypes in latent space
    
    Args:
        x: Input tensor (shape should match what prototype_layer expects)
        prototype_layer: Your model's prototype layer with prototype distances
    
    Returns:
        distances: Tensor of distances to each prototype
    """
    # Forward pass through the model up to prototype layer
    # This depends on your specific model architecture
    latent_x = prototype_layer(x)  # Get latent representation
    
    # Get prototype vectors from the layer
    prototypes = prototype_layer.prototypes  # Shape: (n_prototypes, latent_dim)
    
    # Compute squared L2 distances between x and each prototype
    distances = torch.cdist(latent_x.unsqueeze(0), prototypes.unsqueeze(0), p=2).squeeze()
    
    return distances

def compute_D(prototype_distances, prototype_labels, y):
    """
    Compute D(x,y) - root mean squared distance to prototypes of class y
    h_x: distances from input x to each prototype (n_prototypes,)
    prototype_labels: array of labels for each prototype (n_prototypes,)
    y: target class
    """
    mask = (prototype_labels == y)
    squared_distances = prototype_distances[mask]**2


    return np.sqrt(np.mean(squared_distances)) if np.any(mask) else 0.0

def compute_R(prototype_distances, prototype_labels, y):
    """
    Compute R(x,y) - root mean squared distance to prototypes not of class y
    prototype_distances: distances from input x to each prototype (n_prototypes,)
    prototype_labels: array of labels for each prototype (n_prototypes,)
    y: target class
    """
    mask = (prototype_labels != y)
    squared_distances = prototype_distances[mask]**2
    return np.sqrt(np.mean(squared_distances)) if np.any(mask) else 0.0

def L_h(prototype_distances, prototype_labels, lambda_param):
    """
    Compute explanation loss L_h(x,y) = D(x,y) - λR(x,y)
    """
    D = compute_D(prototype_distances, prototype_labels)
    R = compute_R(prototype_distances, prototype_labels)
    return D - lambda_param * R

def objective_function(model, x, xi, alpha, lambda_param, epsilon):
    """
    Compute the full objective function to maximize:
    L_h(x*, y) - ξL_y(f(x*), y) - α||h(x*)||₂
    
    Parameters:4
    """
    
    # Get distances to prototypes and class probabilities
    prototype_distances = model.prototype_layer.prototype_distances # Latent representation of the prototypes
    prototype_labels = np.where((model.fc.linear.weight.cpu().detach().numpy().T <= np.min(model.fc.linear.weight.cpu().detach().numpy().T, axis=1, keepdims=True)) | (model.fc.linear.weight.cpu().detach().numpy().T < thresh), 1, 0)
    
    # Compute each term
    explanation_loss = L_h(prototype_distances, prototype_labels, lambda_param)
    class_loss = torch.nn.CrossEntropyLoss()  # Cross-entropy loss
    stability_term = alpha * np.linalg.norm(prototype_distances)
    
    return explanation_loss - xi * class_loss - stability_term

# Example usage for optimization:
def find_optimal_perturbation(h, f, x_original, y_true, prototype_labels,
                            xi=0.5, alpha=0.1, lambda_param=0.7, epsilon=0.1):
    """
    Find x* that maximizes the objective function within ε-ball of x_original
    """
    from scipy.optimize import minimize
    
    bounds = [(xi - epsilon, xi + epsilon) for xi in x_original.flatten()]
    
    result = minimize(
        lambda x: -objective_function(
            x.reshape(x_original.shape), h, f, prototype_labels, y_true,
            xi, alpha, lambda_param, epsilon, x_original
        ),
        x0=x_original.flatten(),
        bounds=bounds,
        method='L-BFGS-B'
    )
    
    return result.x.reshape(x_original.shape)

# Example prototype setup (would need your actual prototypes)
"""
# Assuming:
# - 10 prototypes in 20-dimensional space
prototypes = np.random.randn(10, 20)  
prototype_labels = np.random.randint(0, 3, 10)  # 3 classes
h = lambda x: np.array([np.linalg.norm(x - p) for p in prototypes])  # distance function
f = lambda x: softmax(np.dot(x, prototype_weights.T))  # dummy classifier
x_original = np.random.randn(20)
y_true = 1  # true class

optimal_x_star = find_optimal_perturbation(h, f, x_original, y_true, prototype_labels)
"""

def adversarial_explanation_objective(model, x_star, x_original, y, xi=0.5, alpha=0.1, epsilon=0.1):
    """
    Implements the adversarial explanation objective:
    arg max L_h(x*, y) - ξL_y(f(x*), y) - α||h(x*)||₂
    subject to x* ∈ B_ε(x)
    
    Args:
        model: The model containing prototype layer and classifier
        x_star: The perturbed input we're optimizing (B x C x H x W)
        x_original: The original input (B x C x H x W)
        y: True class label (B,)
        xi: Weight for classification loss (default 0.5)
        alpha: Weight for stability term (default 0.1)
        epsilon: Size of the allowed perturbation ball (default 0.1)
        
    Returns:
        total_loss: The objective value to be maximized
    """
    device = x_star.device
    batch_size = x_star.shape[0]
    
    # Ensure input is within epsilon ball of original
    delta = x_star - x_original
    norm = torch.norm(delta.view(batch_size, -1), dim=1)
    mask = norm > epsilon
    if mask.any():
        delta[mask] = epsilon * F.normalize(delta[mask].view(mask.sum(), -1), dim=1).view(delta[mask].shape)
        x_star = x_original + delta
    
    # Get model predictions and prototype distances
    with torch.set_grad_enabled(True):
        # Forward pass through the model
        feature_vectors = model.forward_features(x_star)  # Get features before prototype layer
        prototype_distances = model.prototype_layer(feature_vectors)  # Get distances to prototypes
        pred_y = model.forward_logits(feature_vectors)  # Get logits
    
    # Get prototype labels (ensure on correct device)
    prototype_classes = torch.where(
        (model.fc.linear.weight.T <= torch.min(model.fc.linear.weight.T, dim=1, keepdim=True)[0]) | 
        (model.fc.linear.weight.T < thresh),
        torch.ones(1, device=device),
        torch.zeros(1, device=device)
    )
    prototype_labels = torch.where(prototype_classes == 1)[1]
    
    # 1. Explanation loss L_h(x*, y) - using existing prototype_loss function
    explanation_loss = prototype_loss(
        prototype_distances,
        prototype_labels,
        y,
        lambda_param=0.5  # This could be made configurable
    )
    
    # 2. Classification loss L_y(f(x*), y) with numerical stability
    classification_loss = F.cross_entropy(
        pred_y,
        y,
        reduction='mean',
        label_smoothing=0.1  # Add label smoothing for stability
    )
    
    # 3. Stability term ||h(x*)||₂ with improved numerical stability
    stability_term = alpha * torch.norm(prototype_distances.view(batch_size, -1), dim=1).mean()
    
    # Combine terms according to formula with proper scaling
    total_loss = explanation_loss - xi * classification_loss - stability_term
    
    return total_loss

def generate_adversarial_explanation(model, x, y, xi=0.5, alpha=0.1, epsilon=0.1, 
                                  num_steps=100, lr=0.01, early_stopping_patience=10):
    """
    Generates an adversarial example x* that maximizes the adversarial explanation objective.
    
    Args:
        model: The model to attack
        x: Original input (B x C x H x W)
        y: True class label (B,)
        xi: Weight for classification loss
        alpha: Weight for stability term
        epsilon: Maximum perturbation size
        num_steps: Number of optimization steps
        lr: Learning rate
        early_stopping_patience: Number of steps to wait before early stopping
        
    Returns:
        x_star: The generated adversarial example
        loss_history: List of loss values during optimization
    """
    device = x.device
    model.eval()  # Ensure model is in eval mode
    
    # Initialize x_star with small random noise within epsilon ball
    x_star = x.clone().detach()
    noise = torch.randn_like(x_star, device=device) * epsilon * 0.1
    x_star = x_star + noise
    x_star.requires_grad_(True)
    
    # Setup optimizer with gradient clipping
    optimizer = torch.optim.Adam([x_star], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=False
    )
    
    best_loss = float('-inf')
    best_x_star = x_star.clone()
    patience_counter = 0
    loss_history = []
    
    try:
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Compute objective
            loss = adversarial_explanation_objective(model, x_star, x, y, xi, alpha, epsilon)
            
            # Update x_star
            (-loss).backward()  # Negative since we want to maximize
            torch.nn.utils.clip_grad_norm_([x_star], max_norm=1.0)
            optimizer.step()
            
            # Project back to epsilon ball
            with torch.no_grad():
                delta = x_star - x
                norm = torch.norm(delta.view(delta.shape[0], -1), dim=1, keepdim=True)
                factor = torch.min(torch.ones_like(norm), epsilon / norm)
                delta = delta * factor.view(-1, 1, 1, 1)
                x_star.data = x + delta
            
            # Update learning rate
            current_loss = loss.item()
            loss_history.append(current_loss)
            scheduler.step(current_loss)
            
            # Early stopping check
            if current_loss > best_loss:
                best_loss = current_loss
                best_x_star = x_star.clone()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                break
                
    except RuntimeError as e:
        print(f"Optimization stopped due to error: {e}")
        return best_x_star.detach(), loss_history
    
    return best_x_star.detach(), loss_history