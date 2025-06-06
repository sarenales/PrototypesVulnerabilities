import torch
import numpy as np
from functools import partial
from modules import Softmax
# from metrics import *
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from loss_functions import Loss_1

softmax = Softmax()

def show_adversarial_examples(model, batch_x, perturbed_batch_x, batch_y, pred_y, pred_y_adv, conf_y, conf_y_adv, n_examples, examples_type):
    """
    Display adversarial examples along with their closest prototypes.

    Args:
        model (torch.nn.Module): The trained model.
        batch_x (torch.Tensor): The original input batch.
        perturbed_batch_x (torch.Tensor): The perturbed input batch.
        batch_y (torch.Tensor): The true labels for the input batch.
        pred_y (torch.Tensor): The predicted labels for the input batch.
        pred_y_adv (torch.Tensor): The predicted labels for the perturbed input batch.
        conf_y (torch.Tensor): The confidence scores for the true labels in the input batch.
        conf_y_adv (torch.Tensor): The confidence scores for the predicted labels in the perturbed input batch.
        n_examples (int): The maximum number of examples to display.
        examples_type (str): The type of examples to display. Can be one of the following:
            - "cdp": Correctly classified and the closest prototype is different.
            - "csp": Correctly classified and the closest prototype is the same.
            - "isp": Incorrectly classified and the closest prototype is the same.
            - "idp": Incorrectly classified and the closest prototype is different.

    Returns:
        int: The total number of examples shown.
    """
    prototypes_to_show = 15
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Define the device
    model.prototype_layer.prototype_distances = model.prototype_layer.prototype_distances.to(device)
    prototype_distances = model.prototype_layer.prototype_distances
    n_prototypes = prototype_distances.size(0)
    prototype_imgs = model.decoder(prototype_distances.reshape((-1,10,2,2))).detach().cpu()

    n_cols = 5
    total_examples = 0  # Counter for total number of examples shown

    # Distances for normal and adversarial examples 
    distances = model.prototype_layer(model.encoder(batch_x).view(batch_x.size(0), -1))  
    distances_adv = model.prototype_layer(model.encoder(perturbed_batch_x).view(perturbed_batch_x.size(0), -1)) 
    for idx in range(batch_x.size(0)): 

        if n_examples == total_examples:
            break

        dists = distances[idx].detach().cpu().numpy()  

        # Sort prototypes by distances and keep only the closest 10
        sorted_prototypes = sorted(zip(prototype_imgs, dists), key=lambda x: x[1])[:prototypes_to_show]
        sorted_prototype_imgs, sorted_dists = zip(*sorted_prototypes)

        input_img_adv = perturbed_batch_x[idx].detach().cpu().numpy().reshape(batch_x.size(-2), batch_x.size(-1))
        dists_adv = distances_adv[idx].detach().cpu().numpy()  

        # Sort prototypes by distances for adversarial examples and keep only the closest 10
        sorted_prototypes_adv = sorted(zip(prototype_imgs, dists_adv), key=lambda x: x[1])[:prototypes_to_show]
        sorted_prototype_imgs_adv, sorted_dists_adv = zip(*sorted_prototypes_adv)

        if examples_type == "cdp":
            cond = pred_y_adv[idx] == batch_y[idx] and not torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0])
        elif examples_type == "csp":
            cond = pred_y_adv[idx] == batch_y[idx] and torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0])
        elif examples_type == "isp":
            cond = pred_y_adv[idx] != batch_y[idx] and torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0])
        elif examples_type == "idp":
            cond = pred_y_adv[idx] != batch_y[idx] and not torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0])

        if cond: 

            input_img = batch_x[idx].detach().cpu().numpy().reshape(batch_x.size(-2), batch_x.size(-1))

            gs = gridspec.GridSpec(1, 2, width_ratios=[1, n_cols], wspace=0.1)  

            ax0 = plt.subplot(gs[0])
            ax0.imshow(input_img, cmap='gray', interpolation='none')
            ax0.set_title("Standard")
            ax0.axis('off')
            ax0.text(0.5, -0.1, 'Y: {:.2f}  y: {:.2f}'.format(batch_y[idx], pred_y[idx]), horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes, fontsize=10)
            ax0.text(0.5, -0.4, 'Conf: {:.2f}'.format(conf_y[idx]), horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes, fontsize=10)

            # Calculate the number of rows needed for this image
            n_rows = np.ceil(prototypes_to_show / n_cols).astype(int)

            gs1 = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs[1], wspace=0.1, hspace=0.1)

            for p_idx in range(prototypes_to_show):  # Only show the closest 10 prototypes
                row = p_idx // n_cols
                col = p_idx % n_cols
                ax = plt.subplot(gs1[row, col])
                prototype_img = sorted_prototype_imgs[p_idx].detach().cpu().numpy().reshape(batch_x.size(-2), batch_x.size(-1))
                ax.imshow(prototype_img, cmap='gray', interpolation='none')

                ax.text(0.5, -0.1, f'{sorted_dists[p_idx]:.2f}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
                ax.axis('off')

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Further adjust space between plots
            plt.show()

            gs = gridspec.GridSpec(1, 2, width_ratios=[1, n_cols], wspace=0.1)  

            ax0 = plt.subplot(gs[0])
            ax0.imshow(input_img_adv, cmap='gray', interpolation='none')
            ax0.set_title("Adversarial")
            ax0.axis('off')
            ax0.text(0.5, -0.1, 'Y: {:.2f}  y: {:.2f}'.format(batch_y[idx], pred_y_adv[idx]), horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes, fontsize=10)
            ax0.text(0.5, -0.4, 'Conf: {:.2f}'.format(conf_y_adv[idx]), horizontalalignment='center', verticalalignment='center', transform=ax0.transAxes, fontsize=10)

            n_rows = np.ceil(prototypes_to_show / n_cols).astype(int)

            gs1 = gridspec.GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=gs[1], wspace=0.1, hspace=0.1)

            for p_idx in range(prototypes_to_show):  # Only show the closest 15 prototypes
                row = p_idx // n_cols
                col = p_idx % n_cols
                ax = plt.subplot(gs1[row, col])
                prototype_img = sorted_prototype_imgs_adv[p_idx].detach().cpu().numpy().reshape(batch_x.size(-2), batch_x.size(-1))
                ax.imshow(prototype_img, cmap='gray', interpolation='none')

                ax.text(0.5, -0.1, f'{sorted_dists_adv[p_idx]:.2f}', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=10)
                ax.axis('off')

            plt.tight_layout()
            plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Further adjust space between plots
            plt.show()
                
            total_examples += 1  # Increment the total examples counter

        #if total_examples >= n_examples:  # Stop when the number of examples have been achieved
            break

    return total_examples

def attack_effects_test(model, test_loader, loss, attack):
    """
    Evaluates the effects of an attack on a given model using a test dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        test_loader (torch.utils.data.DataLoader): The test dataset loader.
        loss (torch.nn.Module): The loss function to use for evaluation.
        attack (function): The attack function to generate adversarial examples.

    Returns:
        tuple: A tuple containing the test accuracy, adversarial test accuracy, and metric percentage.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device
    test_ac = 0
    test_ac_adv = 0
    n_test_batch = len(test_loader)
    
    objective = loss.keywords['objective']
    force_class = loss.keywords['force_class']
    change_expl = loss.keywords['change_expl']
    alpha1 = loss.keywords['alpha1']
    alpha2 = loss.keywords['alpha2']
    
    metric_percentage = 0
    
    thresh = -1.8 # We want to find prototypes for all the classes
    prototype_classes = np.where((model.fc.linear.weight.cpu().detach().numpy().T <= np.min(model.fc.linear.weight.cpu().detach().numpy().T, axis=1, keepdims=True)) | (model.fc.linear.weight.cpu().detach().numpy().T < thresh), 1, 0)
    
    for i, batch in enumerate(test_loader):
        batch_x = batch[0]
        batch_y = batch[1]
        batch_x = batch_x.to(device)
        batch_x.requires_grad = True
        batch_y = batch_y.to(device)

        # Get the predictions for the non-adversarial examples
        pred_y = model.forward(batch_x)
        pred_y = softmax(pred_y)

        # Generate adversarial examples from batch_x
        loss_f = partial(loss, batch_y=batch_y)
        adv_attack = partial(attack, loss_f=loss_f)
        perturbed_batch_x = adv_attack(batch_x)

        # Get the predictions for the adversarial examples
        pred_y_adv = model.forward(perturbed_batch_x)
        pred_y_adv = softmax(pred_y_adv)

        # test accuracy
        conf_y, max_indices = torch.max(pred_y,1)
        n = max_indices.size(0)
        test_ac += (max_indices == batch_y).sum(dtype=torch.float32)/n

        #adversarial test accuracy
        conf_y_adv, max_indices_adv = torch.max(pred_y_adv,1)
        n = max_indices_adv.size(0)
        test_ac_adv += (max_indices_adv == batch_y).sum(dtype=torch.float32)/n

        # Distances to prototypes for normal and adversarial examples 
        distances = model.prototype_layer(model.encoder(batch_x).view(batch_x.size(0), -1))  
        distances_adv = model.prototype_layer(model.encoder(perturbed_batch_x).view(perturbed_batch_x.size(0), -1)) 

        # For each example in the batch, verify if the attack has been effective or not
        for idx in range(batch_x.size(0)): 

            dists = distances[idx].detach().cpu().numpy()  
            dists_adv = distances_adv[idx].detach().cpu().numpy()  

            sorted_prototype_classes = sorted(zip(prototype_classes, dists), key=lambda x: x[1])
            sorted_prototype_classes_adv = sorted(zip(prototype_classes, dists_adv), key=lambda x: x[1])
            
            #Class of the closest prototype
            explanation_class = np.where(sorted_prototype_classes[0][0] == 1)[0][0]
            explanation_class_adv = np.where(sorted_prototype_classes_adv[0][0] == 1)[0][0]
            
            #If objective is change class and dont care about explanation
            if objective == "cecc" or objective == "necc" and alpha1 == 1 and alpha2 == 0:
                #If it was correctly classified before and now badly
                if  batch_y[idx] == max_indices[idx] and batch_y[idx] != max_indices_adv[idx]:
                    
                    if force_class is None:
                        metric_percentage += 1/n
                        
                    elif force_class is not None and force_class == max_indices_adv[idx]:
                        metric_percentage += 1/n
                        
            #If objective is change explanation and dont care about class
            elif objective == "cecc" or objective == "cenc" and alpha1 == 0 and alpha2 == 1:
                #If the explanation class has changed
                if explanation_class != explanation_class_adv:
                    
                    if change_expl is None:
                        metric_percentaje += 1/n
                        
                    elif change_expl is not None and explanation_class_adv == change_expl:
                        metric_percentage += 1/n
                
            elif objective == "cecc" and alpha1 == 1 and alpha2 == 1:
                #If the explanation class has changed and it was correctly classified before and now badly
                if explanation_class != explanation_class_adv and batch_y[idx] == max_indices[idx] \
                    and batch_y[idx] != max_indices_adv[idx]:
                    
                    if change_expl is None and force_class is None:
                        metric_percentaje += 1/n
                    
                    elif change_expl is not None and force_class is None \
                        and force_class == max_indices_adv[idx] and explanation_class_adv == change_expl:
                        metric_percentage += 1/n
            else:
                raise ValueError("You are not using a logic configuration, check")
                
    test_ac /= n_test_batch
    test_ac_adv /= n_test_batch
    metric_percentage /= n_test_batch 
    
    return test_ac, test_ac_adv, metric_percentage
    
def run_tests(model, test_loader, attack, loss_generator, num_runs=5):
    """
    Run multiple tests on a model using a given test loader and attack method.

    Args:
        model (torch.nn.Module): The model to be tested.
        test_loader (torch.utils.data.DataLoader): The data loader for the test set.
        attack (function): The attack method to be used.
        loss_generator (function): A function that generates the loss for the attack.
        num_runs (int, optional): The number of times to run the tests. Defaults to 5.

    Returns:
        tuple: A tuple containing three sub-tuples:
            - (test_ac_mean, test_ac_std): Mean and standard deviation of accuracy on the default test set.
            - (test_ac_adv_mean, test_ac_adv_std): Mean and standard deviation of accuracy on the adversarial test set.
            - (metric_percentage_mean, metric_percentage_std): Mean and standard deviation of the effectiveness percentage for the adversarial test set.
    """
    test_ac_list = []
    test_ac_adv_list = []
    metric_percentage_list = []
    
    # Run the tests num_runs times
    for _ in range(num_runs):
        loss = loss_generator()
        test_ac, test_ac_adv, metric_percentage = attack_effects_test(model, test_loader, attack=attack, loss=loss)
        test_ac_list.append(test_ac.cpu().numpy())
        test_ac_adv_list.append(test_ac_adv.cpu().numpy())
        metric_percentage_list.append(metric_percentage)  
    
    # Calculate the mean and standard deviation for the default test set
    test_ac_mean = np.mean(test_ac_list)
    test_ac_std = np.std(test_ac_list)
    
    # Calculate the mean and standard deviation for the adversarial test set
    test_ac_adv_mean = np.mean(test_ac_adv_list)
    test_ac_adv_std = np.std(test_ac_adv_list)
    
    # Calculate the mean and standard deviation for the effectiveness percentage for the adversarial test set
    metric_percentage_mean = np.mean(metric_percentage_list)
    metric_percentage_std = np.std(metric_percentage_list)
    
    return (test_ac_mean, test_ac_std), (test_ac_adv_mean, test_ac_adv_std), (metric_percentage_mean, metric_percentage_std)

def adversarial_attacks_eps_plot(models, model_names, test_loader, attack, loss, max_eps, step=0.0001):
    """
    Plots the accuracy of models under adversarial attacks for different epsilon values.

    Args:
        models (list): A list of PyTorch models.
        model_names (list): A list of names corresponding to the models.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        attack (function): The adversarial attack function.
        loss (function): The loss function used for the attack.
        max_eps (int or float): The maximum epsilon value for the attack.
        step (float, optional): The step size for epsilon values. Defaults to 0.01.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device
    dim = max_eps+1 if isinstance(max_eps, int) else int(max_eps/step) + 1
   
    results = np.zeros((len(models), dim))

    for i, batch in enumerate(test_loader):
        batch_x = batch[0]
        batch_y = batch[1]
        batch_x = batch_x.to(device)
        batch_x.requires_grad = True
        batch_y = batch_y.to(device)

        for idm, model in enumerate(models):
            pred_y = model.forward(batch_x)
            pred_y = softmax(pred_y)

            loss_f = partial(loss, model=model, batch_y=batch_y)

            # Non-adversarial test set accuracy
            conf_y, max_indices = torch.max(pred_y,1)
            n = max_indices.size(0)
            results[idm, 0] += (max_indices == batch_y).sum(dtype=torch.float32)/n
            
            # Adversarial test set accuracy for different epsilon values
            maxx = max_eps if isinstance(max_eps, int) else 1000 * max_eps
            s = 1 if isinstance(max_eps, int) else step * 1000
            for ide, epss in enumerate(np.arange(s, maxx+s, s)):
                
                # Convert epsilon to the right scale
                eps = epss / 1000 if not isinstance(max_eps, int) else epss
                
                # Generate adversarial examples from batch_x for the corresponding epsilon
                adv_attack = partial(attack, loss_f=loss_f, eps=eps)
                perturbed_batch_x = adv_attack(batch_x)

                # Get the predictions for the adversarial examples
                pred_y_adv = model.forward(perturbed_batch_x)

                #adversarial test accuracy
                conf_y_adv, max_indices_adv = torch.max(pred_y_adv,1)
                n = max_indices_adv.size(0)
                results[idm, ide+1] += (max_indices_adv == batch_y).sum(dtype=torch.float32)/n

    results /= len(test_loader)

    x_axis = np.arange(0, max_eps+step, step) if isinstance(max_eps, float) else np.arange(0, max_eps+1, 1)

    for i in range(len(models)):
        plt.plot(x_axis, results[i], label=model_names[i])
        plt.scatter(x_axis, results[i])

    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
 
def test_model(model, test_loader):
    """
    Test the given model on the test dataset and calculate the accuracy.

    Args:
        model (torch.nn.Module): The model to be tested.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device
    test_ac = 0.0

    for i, batch in enumerate(test_loader):
        batch_x = batch[0]
        batch_y = batch[1]
        batch_x = batch_x.to(device)
        batch_x.requires_grad = True
        batch_y = batch_y.to(device)


        pred_y = model.forward(batch_x)

        # test accuracy
        conf_y, max_indices = torch.max(pred_y,1)
        n = max_indices.size(0)
        test_ac += (max_indices == batch_y).sum(dtype=torch.float32)/n

    test_ac /= len(test_loader)

    print("test set accuracy: {:.4f}".format(test_ac))

def draw_confusion_matrix(cm, categories):
    """
    Draw a confusion matrix for adversarial examples.
    
    Args:
        cm (numpy.ndarray): The confusion matrix
        categories (dict): Dictionary mapping class indices to class names
    """
    fig = plt.figure(figsize=[6.4*pow(len(categories), 0.5), 4.8*pow(len(categories), 0.5)])
    ax = fig.add_subplot(111)
    cm = cm.astype('float') / np.maximum(cm.sum(axis=1)[:, np.newaxis], np.finfo(np.float64).eps)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.get_cmap('Blues'))
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), 
           yticks=np.arange(cm.shape[0]), 
           xticklabels=list(categories.values()), 
           yticklabels=list(categories.values()), 
           ylabel='True Label', 
           xlabel='Predicted Label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], '.2f'), 
                   ha="center", va="center", 
                   color="white" if cm[i, j] > thresh else "black", 
                   fontsize=int(20-pow(len(categories), 0.5)))
    fig.tight_layout()
    plt.show()


def test_adversarial(model, test_loader, loss, attack, n_examples, examples_type):
    """
    Test the model's performance on adversarial examples.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        loss (callable): The loss function used for training the model.
        attack (callable): The attack function used to generate adversarial examples.
        n_examples (int): The maximum number of adversarial examples to show.
        examples_type (str): The type of adversarial examples to show. Can be one of the following:
            - "cdp": Correctly classified and the closest prototype is different.
            - "csp": Correctly classified and the closest prototype is the same.
            - "isp": Incorrectly classified and the closest prototype is the same.
            - "idp": Incorrectly classified and the closest prototype is different.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Define the device
    test_ac = 0
    test_ac_adv = 0
    n_test_batch = len(test_loader)

    corr_dist_proto = 0
    corr_same_proto = 0
    incorr_same_proto = 0
    incorr_dist_proto = 0


    prototype_distances = model.prototype_layer.prototype_distances
    prototype_imgs = model.decoder(prototype_distances.reshape((-1,10,2,2))).detach().cpu()
    total_examples = 0

    for i, batch in enumerate(test_loader):
        batch_x = batch[0]
        batch_y = batch[1]
        batch_x = batch_x.to(device)
        batch_x.requires_grad = True
        batch_y = batch_y.to(device)

        pred_y = model.forward(batch_x)
        pred_y = softmax(pred_y)

        loss_f = partial(loss, batch_y=batch_y)
        adv_attack = partial(attack, loss_f=loss_f)

        perturbed_batch_x = adv_attack(batch_x)

        pred_y_adv = model.forward(perturbed_batch_x)
        pred_y_adv = softmax(pred_y_adv)

        # test accuracy
        conf_y, max_indices = torch.max(pred_y,1)
        n = max_indices.size(0)
        test_ac += (max_indices == batch_y).sum(dtype=torch.float32)/n

        #adversarial test accuracy
        conf_y_adv, max_indices_adv = torch.max(pred_y_adv,1)
        n = max_indices_adv.size(0)
        test_ac_adv += (max_indices_adv == batch_y).sum(dtype=torch.float32)/n

        total_examples += show_adversarial_examples(model, batch_x, perturbed_batch_x, batch_y, max_indices, max_indices_adv, conf_y, conf_y_adv, n_examples-total_examples, examples_type)

        # Distances for normal and adversarial examples 
        distances = model.prototype_layer(model.encoder(batch_x).view(batch_x.size(0), -1))  
        distances_adv = model.prototype_layer(model.encoder(perturbed_batch_x).view(perturbed_batch_x.size(0), -1)) 

        for idx in range(batch_x.size(0)): 

            dists = distances[idx].detach().cpu().numpy()  
            dists_adv = distances_adv[idx].detach().cpu().numpy()  

            # Sort prototypes by distances
            sorted_prototypes = sorted(zip(prototype_imgs, dists), key=lambda x: x[1])
            sorted_prototype_imgs, sorted_dists = zip(*sorted_prototypes)

            # Sort prototypes by distances for adversarial examples
            sorted_prototypes_adv = sorted(zip(prototype_imgs, dists_adv), key=lambda x: x[1])
            sorted_prototype_imgs_adv, sorted_dists_adv = zip(*sorted_prototypes_adv)

            # If correctly classified and the closest prototype is different
            if batch_y[idx] == max_indices_adv[idx] and not torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0]):
                corr_dist_proto += 1/n

            # If incorrectly classified and the closest prototype is the same
            elif batch_y[idx] != max_indices_adv[idx] and torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0]):
                incorr_same_proto += 1/n

            # If correctly classified and the closest prototype is the same
            elif batch_y[idx] == max_indices_adv[idx] and torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0]):
                corr_same_proto += 1/n

            #If incorrectly classified and the closest prototype is different
            elif batch_y[idx] != max_indices_adv[idx] and not torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0]):
                incorr_dist_proto += 1/n

    test_ac /= n_test_batch
    test_ac_adv /= n_test_batch
    corr_dist_proto /= n_test_batch 
    corr_same_proto /= n_test_batch
    incorr_same_proto /= n_test_batch
    incorr_dist_proto /= n_test_batch

    print("test set:")
    print("\taccuracy: {:.4f}".format(test_ac))

    print("adversarial test set:")
    print("\taccuracy: {:.4f}".format(test_ac_adv))
    print("\tCorrectly classified and the closest prototype is different: {:.4f}".format(corr_dist_proto))
    print("\tCorrectly classified and the closest prototype is the same: {:.4f}".format(corr_same_proto))
    print("\tIncorrectly classified and the closest prototype is the same: {:.4f}".format(incorr_same_proto))
    print("\tIncorrectly classified and the closest prototype is different: {:.4f}".format(incorr_dist_proto))




def get_encoded_test_data_and_fit_pca(test_loader, model, device):
    """
    Encodes test data using the model's encoder and fits a PCA transformation on the encoded data.

    Args:
    - test_loader (DataLoader): DataLoader for the test dataset.
    - model (nn.Module): The model containing the encoder.
    - device (torch.device): The device to run the model on (CPU or GPU).

    Returns:
    - reduced_test_data (np.ndarray): 2D PCA projection of the encoded test data.
    - labels (np.ndarray): Labels of the test data.
    - pca (PCA): Fitted PCA object.
    """
    encoded_data = []
    labels = []

    # Encode the test data
    for batch in test_loader:
        batch_x, batch_y = batch
        batch_x = batch_x.to(device)
        encoded_batch = model.encoder(batch_x).detach().cpu().numpy().reshape(batch_x.size(0), -1)
        encoded_data.append(encoded_batch)
        labels.append(batch_y.numpy())

    encoded_data = np.concatenate(encoded_data, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Fit PCA on the encoded data
    pca = PCA(n_components=2)
    reduced_test_data = pca.fit_transform(encoded_data)

    return reduced_test_data, labels, pca

def get_prototype_projection(model, pca):
    """
    Projects the prototypes of the model into a 2D PCA space.

    Args:
    - model_path (str): Path to the model file.
    - device (torch.device): The device to run the model on (CPU or GPU).
    - pca (PCA): Fitted PCA object.

    Returns:
    - reduced_prototypes (np.ndarray): 2D PCA projection of the prototypes.
    - prototype_imgs (torch.Tensor): Decoded prototype images.
    """
    #model = torch.load(model_path, weights_only=False)
    #model.to(device)
    model.prototype_layer.prototype_distances = model.prototype_layer.prototype_distances
    model.eval()

    prototype_distances = model.prototype_layer.prototype_distances
    prototype_imgs = model.decoder(prototype_distances.reshape((-1, 10, 2, 2))).detach().cpu()

    # Project prototypes using the fitted PCA
    reduced_prototypes = pca.transform(prototype_distances.detach().cpu().numpy().reshape(-1, 40))

    return reduced_prototypes, prototype_imgs

def plot_prototype_projection_with_data(reduced_prototypes, prototype_imgs, reduced_test_data, test_labels, title, xlim, ylim, save_path):
    """
    Plots the 2D PCA projection of prototypes and test data.

    Args:
    - reduced_prototypes (np.ndarray): 2D PCA projection of the prototypes.
    - prototype_imgs (torch.Tensor): Decoded prototype images.
    - reduced_test_data (np.ndarray): 2D PCA projection of the test data.
    - test_labels (np.ndarray): Labels of the test data.
    - title (str): Title of the plot.
    - xlim (tuple): x-axis limits for the plot.
    - ylim (tuple): y-axis limits for the plot.
    - save_path (str): Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8), dpi=300)
    ax.set_title(title)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')

    image_size = 0.2  # Adjust this value to make images smaller

    # Plot the encoded test data with different colors for each class using 'tab20' colormap
    scatter = ax.scatter(reduced_test_data[:, 0], reduced_test_data[:, 1], c=test_labels, cmap='tab20', alpha=0.5)
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)

    # Overlay prototype images
    for i, (x, y) in enumerate(reduced_prototypes):
        img = prototype_imgs[i].squeeze().numpy()  # Remove single-dimensional entries
        ax.imshow(img, cmap='gray', extent=(x - image_size / 2, x + image_size / 2, y - image_size / 2, y + image_size / 2), aspect='auto', zorder=10)
        ax.scatter(x, y, c='red', s=1, zorder=11)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # plt.savefig(save_path, bbox_inches='tight')
    #plt.close(fig)
    plt.show()





def test_adversarial_2(model, test_loader, loss, attack, n_examples, examples_type):
    """
    Test the model's performance on adversarial examples and visualize results.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        loss (callable): The loss function used for training the model.
        attack (callable): The attack function used to generate adversarial examples.
        n_examples (int): The maximum number of adversarial examples to show.
        examples_type (str): The type of adversarial examples to show.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ac = 0
    test_ac_adv = 0
    n_test_batch = len(test_loader)

    corr_dist_proto = 0
    corr_same_proto = 0
    incorr_same_proto = 0
    incorr_dist_proto = 0

    prototype_distances = model.prototype_layer.prototype_distances
    prototype_imgs = model.decoder(prototype_distances.reshape((-1,10,2,2))).detach().cpu()
    total_examples = 0

    # Initialize counters for evaluation metrics
    correct_clean = 0
    correct_adv = 0
    total = 0

    # Lists to store predictions for confusion matrix
    all_true_labels = []
    all_clean_preds = []
    all_adv_preds = []

    # Visualization setup
    plt.figure(figsize=(15, 5))
    vis_count = 0

    for i, batch in enumerate(test_loader):
        batch_x = batch[0]
        batch_y = batch[1]
        batch_x = batch_x.to(device)
        batch_x.requires_grad = True
        batch_y = batch_y.to(device)

        # Handle channel conversion if needed
        if batch_x.shape[1] == 3:
            grayscale_images = 0.2989 * batch_x[:,0] + 0.5870 * batch_x[:,1] + 0.1140 * batch_x[:,2]
            grayscale_images = grayscale_images.unsqueeze(1)
        else:
            grayscale_images = batch_x

        pred_y = model.forward(grayscale_images)
        pred_y = softmax(pred_y)

        loss_f = partial(loss, batch_y=batch_y)
        adv_attack = partial(attack, loss_f=loss_f)

        perturbed_batch_x = adv_attack(grayscale_images)

        pred_y_adv = model.forward(perturbed_batch_x)
        pred_y_adv = softmax(pred_y_adv)

        # test accuracy
        conf_y, max_indices = torch.max(pred_y,1)
        n = max_indices.size(0)
        test_ac += (max_indices == batch_y).sum(dtype=torch.float32)/n
        correct_clean += (max_indices == batch_y).sum().item()

        #adversarial test accuracy
        conf_y_adv, max_indices_adv = torch.max(pred_y_adv,1)
        n = max_indices_adv.size(0)
        test_ac_adv += (max_indices_adv == batch_y).sum(dtype=torch.float32)/n
        correct_adv += (max_indices_adv == batch_y).sum().item()
        
        total += batch_y.size(0)

        # Store predictions for confusion matrix
        all_true_labels.extend(batch_y.cpu().numpy())
        all_clean_preds.extend(max_indices.cpu().numpy())
        all_adv_preds.extend(max_indices_adv.cpu().numpy())

        # Visualize results for first batch
        if i == 0:
            for j in range(min(3, len(grayscale_images))):
                # Original image
                plt.subplot(3, 4, j*4+1)
                plt.imshow(grayscale_images[j].detach().squeeze().cpu().numpy(), cmap='gray')
                plt.title(f"Clean\nTrue: {batch_y[j].item()}\nPred: {max_indices[j].item()}")
                plt.axis('off')
                
                # Adversarial image
                plt.subplot(3, 4, j*4+2)
                plt.imshow(perturbed_batch_x[j].detach().squeeze().cpu().numpy(), cmap='gray')
                plt.title(f"Adversarial\nPred: {max_indices_adv[j].item()}")
                plt.axis('off')
                
                # Perturbation
                plt.subplot(3, 4, j*4+3)
                perturbation = (perturbed_batch_x[j] - grayscale_images[j]).detach().abs().squeeze().cpu().numpy()
                plt.imshow(perturbation, cmap='hot')
                plt.title("Perturbation")
                plt.colorbar()
                plt.axis('off')
                
                # Histogram of perturbation
                plt.subplot(3, 4, j*4+4)
                plt.hist(perturbation.flatten(), bins=50)
                plt.title("Perturbation Distribution")
            
            plt.tight_layout()
            plt.show()

        total_examples += show_adversarial_examples(model, grayscale_images, perturbed_batch_x, batch_y, max_indices, max_indices_adv, conf_y, conf_y_adv, n_examples-total_examples, examples_type)

        # Distances for normal and adversarial examples 
        distances = model.prototype_layer(model.encoder(grayscale_images).view(grayscale_images.size(0), -1))  
        distances_adv = model.prototype_layer(model.encoder(perturbed_batch_x).view(perturbed_batch_x.size(0), -1)) 

        for idx in range(grayscale_images.size(0)): 
            dists = distances[idx].detach().cpu().numpy()  
            dists_adv = distances_adv[idx].detach().cpu().numpy()  

            # Sort prototypes by distances
            sorted_prototypes = sorted(zip(prototype_imgs, dists), key=lambda x: x[1])
            sorted_prototype_imgs, sorted_dists = zip(*sorted_prototypes)

            # Sort prototypes by distances for adversarial examples
            sorted_prototypes_adv = sorted(zip(prototype_imgs, dists_adv), key=lambda x: x[1])
            sorted_prototype_imgs_adv, sorted_dists_adv = zip(*sorted_prototypes_adv)

            # If correctly classified and the closest prototype is different
            if batch_y[idx] == max_indices_adv[idx] and not torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0]):
                corr_dist_proto += 1/n

            # If incorrectly classified and the closest prototype is the same
            elif batch_y[idx] != max_indices_adv[idx] and torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0]):
                incorr_same_proto += 1/n

            # If correctly classified and the closest prototype is the same
            elif batch_y[idx] == max_indices_adv[idx] and torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0]):
                corr_same_proto += 1/n

            #If incorrectly classified and the closest prototype is different
            elif batch_y[idx] != max_indices_adv[idx] and not torch.allclose(sorted_prototype_imgs_adv[0], sorted_prototype_imgs[0]):
                incorr_dist_proto += 1/n

    # Create confusion matrices
    categories = {i: str(i) for i in range(10)}  # Assuming 10 classes, modify if different
    cm_clean = confusion_matrix(all_true_labels, all_clean_preds, labels=list(categories.keys()))
    cm_adv = confusion_matrix(all_true_labels, all_adv_preds, labels=list(categories.keys()))

    # Draw confusion matrices
    print("\nConfusion Matrix for Clean Examples:")
    draw_confusion_matrix(cm_clean, categories)
    print("\nConfusion Matrix for Adversarial Examples:")
    draw_confusion_matrix(cm_adv, categories)

    # Calculate final metrics
    clean_acc = 100 * correct_clean / total
    adv_acc = 100 * correct_adv / total
    attack_success = 100 * (correct_clean - correct_adv) / correct_clean if correct_clean > 0 else 0

    print("\nEvaluation Metrics:")
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"Adversarial Accuracy: {adv_acc:.2f}%")
    print(f"Attack Success Rate: {attack_success:.2f}%")
    
    #print("\nPrototype Analysis:")
    #print("test set:")
    #print("\taccuracy: {:.4f}".format(test_ac))

    #print("adversarial test set:")
    #print("\taccuracy: {:.4f}".format(test_ac_adv))
    #print("\tCorrectly classified and the closest prototype is different: {:.4f}".format(corr_dist_proto))
    #print("\tCorrectly classified and the closest prototype is the same: {:.4f}".format(corr_same_proto))
    #print("\tIncorrectly classified and the closest prototype is the same: {:.4f}".format(incorr_same_proto))
    #print("\tIncorrectly classified and the closest prototype is different: {:.4f}".format(incorr_dist_proto))
    
    # return max_indices, max_indices_adv

def analyze_adversarial_robustness(model, test_loader, attack, loss, param_name, param_range, fixed_params=None, loss_params=None):
    """
    Analyzes model robustness against adversarial attacks with different parameter values.
    
    Args:
        model (torch.nn.Module): The model to test
        test_loader (torch.utils.data.DataLoader): Test data loader
        attack (function): The attack function to use
        loss (function): Loss function for the attack
        param_name (str): Name of the parameter to vary ('eps', 'alpha', 'iters')
        param_range (list): List of values to test for the parameter
        fixed_params (dict, optional): Dictionary of fixed parameters for the attack
        loss_params (dict, optional): Dictionary of parameters for the loss function
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Default parameters for PGDLInf attack
    default_params = {
        'eps': 0.3,
        'alpha': 0.01,
        'iters': 40,
        'random_start': True
    }
    
    # Default parameters for Loss_1
    default_loss_params = {
        'alpha1': 0,
        'alpha2': 1,
        'objective': 'advl',
        'force_class': None,
        'change_expl': None
    }
    
    # Update default parameters with any fixed parameters provided
    if fixed_params:
        default_params.update(fixed_params)
    if loss_params:
        default_loss_params.update(loss_params)
    
    accuracies = []
    
    for param_value in param_range:
        # Create attack parameters dictionary
        attack_params = default_params.copy()
        attack_params[param_name] = param_value
        
        # Initialize metrics
        correct = 0
        total = 0
        
        for batch in test_loader:
            batch_x, batch_y = batch
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            # Generate adversarial examples
            loss_f = partial(loss, 
                           model=model,
                           batch_y=batch_y,
                           alpha1=default_loss_params['alpha1'],
                           alpha2=default_loss_params['alpha2'],
                           objective=default_loss_params['objective'],
                           force_class=default_loss_params['force_class'],
                           change_expl=default_loss_params['change_expl'])
            
            adv_attack = partial(attack, 
                               eps=attack_params['eps'],
                               alpha=attack_params['alpha'],
                               iters=attack_params['iters'],
                               random_start=attack_params['random_start'])
            
            perturbed_batch_x = adv_attack(loss_f=loss_f, batch_x=batch_x)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(perturbed_batch_x)
                _, predicted = torch.max(outputs.data, 1)
                
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = 100 * correct / total
        accuracies.append(accuracy)
        
        print(f"{param_name}={param_value:.4f}: Accuracy = {accuracy:.2f}%")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, accuracies, 'b-o')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy (%)')
    plt.title(f'Model Robustness vs {param_name}')
    plt.grid(True)
    plt.show()
    
    return param_range, accuracies

def compare_attack_parameters(model, test_loader, attack, loss, param_ranges, fixed_params=None, loss_params=None):
    """
    Compares different attack parameters and their effects on model robustness.
    
    Args:
        model (torch.nn.Module): The model to test
        test_loader (torch.utils.data.DataLoader): Test data loader
        attack (function): The attack function to use
        loss (function): Loss function for the attack
        param_ranges (dict): Dictionary of parameter names and their ranges to test
            Example: {'eps': [0.01, 0.05, 0.1], 'alpha': [0.1, 0.2, 0.3]}
        fixed_params (dict, optional): Dictionary of fixed parameters for the attack
        loss_params (dict, optional): Dictionary of parameters for the loss function
    """
    results = {}
    
    for param_name, param_range in param_ranges.items():
        print(f"\nAnalyzing {param_name}...")
        param_values, accuracies = analyze_adversarial_robustness(
            model, test_loader, attack, loss, param_name, param_range, fixed_params, loss_params
        )
        results[param_name] = (param_values, accuracies)
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    for param_name, (values, accuracies) in results.items():
        plt.plot(values, accuracies, 'o-', label=param_name)
    
    plt.xlabel('Parameter Value')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Robustness Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return results



def visualize_adversarials(model, test_loader, attack, alpha2_values):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Obtener un solo batch y una sola imagen
    batch = next(iter(test_loader))
    batch_x = batch[0][:1]  # Tomamos solo la primera imagen
    batch_y = batch[1][:1]
    batch_x = batch_x.to(device)
    batch_x.requires_grad = True
    batch_y = batch_y.to(device)

    # Handle channel conversion if needed
    if batch_x.shape[1] == 3:
        grayscale_images = 0.2989 * batch_x[:,0] + 0.5870 * batch_x[:,1] + 0.1140 * batch_x[:,2]
        grayscale_images = grayscale_images.unsqueeze(1)
    else:
        grayscale_images = batch_x

    # Calcular el número de filas y columnas para la matriz
    n_values = len(alpha2_values) + 1  # +1 para la imagen original
    n_cols = 5  # Número de columnas que quieres
    n_rows = (n_values + n_cols - 1) // n_cols  # Cálculo del número de filas necesarias
    
    # Crear figura para visualización
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    axes = axes.flatten()  # Convertir la matriz de ejes en una lista plana
    
    # Mostrar la imagen original en la primera posición
    axes[0].imshow(grayscale_images[0].detach().cpu().squeeze(), cmap='gray')
    axes[0].set_title(f'Original\nTrue: {batch_y[0].item()}')
    axes[0].axis('off')
    
    # Generar y mostrar ejemplos adversarios para cada alpha2
    for j, alpha2 in enumerate(alpha2_values):
        # Configurar el ataque con el alpha2 actual
        loss_f = partial(Loss_1, 
                        model=model, 
                        batch_y=batch_y,
                        alpha1=1, 
                        alpha2=alpha2, 
                        objective="advl", 
                        force_class=None, 
                        change_expl=None)
        
        adv_attack = partial(attack, loss_f=loss_f)
        perturbed_batch_x = adv_attack(grayscale_images)

        # Get predictions
        pred_y_adv = model.forward(perturbed_batch_x)
        pred_y_adv = softmax(pred_y_adv)
        conf_y_adv, max_indices_adv = torch.max(pred_y_adv, 1)

        # Visualizar ejemplo adversarial
        axes[j+1].imshow(perturbed_batch_x[0].detach().cpu().squeeze(), cmap='gray')
        axes[j+1].set_title(f'α2={alpha2}\nTrue: {batch_y[0].item()}\nPred: {max_indices_adv[0].item()}\nConf: {conf_y_adv[0]:.2f}')
        axes[j+1].axis('off')

    # Ocultar los ejes vacíos si los hay
    for j in range(n_values, len(axes)):
        axes[j].axis('off')
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()




def visualize_adversarial_examples_eps(model, test_loader, attack, loss, eps_values, n_examples=3, loss_params=None):
    """
    Visualiza ejemplos adversarios generados con diferentes valores de epsilon.
    
    Args:
        model (torch.nn.Module): El modelo a probar
        test_loader (torch.utils.data.DataLoader): DataLoader para los datos de prueba
        attack (function): Función de ataque a utilizar
        loss (function): Función de pérdida para el ataque
        eps_values (list): Lista de valores de epsilon a probar
        n_examples (int): Número de ejemplos a mostrar
        loss_params (dict, optional): Parámetros para la función de pérdida
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Parámetros por defecto para Loss_1
    default_loss_params = {
        'alpha1': 0,
        'alpha2': 1,
        'objective': 'advl',
        'force_class': None,
        'change_expl': None
    }
    
    if loss_params:
        default_loss_params.update(loss_params)
    
    # Obtener un batch de ejemplos
    batch = next(iter(test_loader))
    batch_x, batch_y = batch
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    
    # Limitar el número de ejemplos
    batch_x = batch_x[:n_examples]
    batch_y = batch_y[:n_examples]
    
    # Crear figura para visualización
    n_rows = n_examples
    n_cols = len(eps_values) + 1  # +1 para la imagen original
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    
    # Mostrar imágenes originales en la primera columna
    for i in range(n_examples):
        axes[i, 0].imshow(batch_x[i].cpu().squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Original\nTrue: {batch_y[i].item()}')
        axes[i, 0].axis('off')
    
    # Generar y mostrar ejemplos adversarios para cada epsilon
    for j, eps in enumerate(eps_values):
        # Configurar el ataque
        loss_f = partial(loss, 
                        model=model,
                        batch_y=batch_y,
                        alpha1=default_loss_params['alpha1'],
                        alpha2=default_loss_params['alpha2'],
                        objective=default_loss_params['objective'],
                        force_class=default_loss_params['force_class'],
                        change_expl=default_loss_params['change_expl'])
        
        adv_attack = partial(attack, 
                           eps=eps,
                           alpha=0.01,
                           iters=40,
                           random_start=True)
        
        # Generar ejemplos adversarios
        perturbed_batch_x = adv_attack(loss_f=loss_f, batch_x=batch_x)
        
        # Obtener predicciones
        with torch.no_grad():
            outputs = model(perturbed_batch_x)
            _, predicted = torch.max(outputs.data, 1)
        
        # Mostrar ejemplos adversarios
        for i in range(n_examples):
            # Calcular la perturbación
            perturbation = (perturbed_batch_x[i] - batch_x[i]).abs().cpu()
            
            # Mostrar imagen adversaria
            axes[i, j+1].imshow(perturbed_batch_x[i].cpu().squeeze(), cmap='gray')
            axes[i, j+1].set_title(f'ε={eps:.3f}\nPred: {predicted[i].item()}')
            axes[i, j+1].axis('off')
            
            # Mostrar la perturbación en un subplot adicional
            if j == len(eps_values) - 1:  # Solo para el último epsilon
                fig_pert, ax_pert = plt.subplots(1, 1, figsize=(3, 3))
                im = ax_pert.imshow(perturbation.squeeze(), cmap='hot')
                ax_pert.set_title(f'Perturbación (ε={eps:.3f})')
                plt.colorbar(im, ax=ax_pert)
                ax_pert.axis('off')
                plt.show()
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar estadísticas de precisión
    print("\nEstadísticas de precisión:")
    for eps in eps_values:
        loss_f = partial(loss, 
                        model=model,
                        batch_y=batch_y,
                        alpha1=default_loss_params['alpha1'],
                        alpha2=default_loss_params['alpha2'],
                        objective=default_loss_params['objective'],
                        force_class=default_loss_params['force_class'],
                        change_expl=default_loss_params['change_expl'])
        
        adv_attack = partial(attack, 
                           eps=eps,
                           alpha=0.01,
                           iters=40,
                           random_start=True)
        
        perturbed_batch_x = adv_attack(loss_f=loss_f, batch_x=batch_x)
        
        with torch.no_grad():
            outputs = model(perturbed_batch_x)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == batch_y).sum().item()
            accuracy = 100 * correct / batch_y.size(0)
            
        print(f"ε={eps:.3f}: Precisión = {accuracy:.2f}%")

def visualize_adversarial_examples_alpha(model, test_loader, attack, loss, alpha_values, eps=0.01, n_examples=3, loss_params=None):
    """
    Visualiza ejemplos adversarios generados con diferentes valores de alpha.
    
    Args:
        model (torch.nn.Module): El modelo a probar
        test_loader (torch.utils.data.DataLoader): DataLoader para los datos de prueba
        attack (function): Función de ataque a utilizar
        loss (function): Función de pérdida para el ataque
        alpha_values (list): Lista de valores de alpha a probar
        eps (float): Valor de epsilon fijo para todos los ataques
        n_examples (int): Número de ejemplos a mostrar
        loss_params (dict, optional): Parámetros para la función de pérdida
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Parámetros por defecto para Loss_1
    default_loss_params = {
        'alpha1': 0,
        'alpha2': 1,
        'objective': 'advl',
        'force_class': None,
        'change_expl': None
    }
    
    if loss_params:
        default_loss_params.update(loss_params)
    
    # Obtener un batch de ejemplos
    batch = next(iter(test_loader))
    batch_x, batch_y = batch
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    
    # Limitar el número de ejemplos
    batch_x = batch_x[:n_examples]
    batch_y = batch_y[:n_examples]
    
    # Crear figura para visualización
    n_rows = n_examples
    n_cols = len(alpha_values) + 1  # +1 para la imagen original
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*3, n_rows*3))
    
    # Mostrar imágenes originales en la primera columna
    for i in range(n_examples):
        axes[i, 0].imshow(batch_x[i].cpu().squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Original\nTrue: {batch_y[i].item()}')
        axes[i, 0].axis('off')
    
    # Generar y mostrar ejemplos adversarios para cada alpha
    for j, alpha in enumerate(alpha_values):
        # Configurar el ataque
        loss_f = partial(loss, 
                        model=model,
                        batch_y=batch_y,
                        alpha1=default_loss_params['alpha1'],
                        alpha2=default_loss_params['alpha2'],
                        objective=default_loss_params['objective'],
                        force_class=default_loss_params['force_class'],
                        change_expl=default_loss_params['change_expl'])
        
        adv_attack = partial(attack, 
                           eps=eps,
                           alpha=alpha,
                           iters=40,
                           random_start=True)
        
        # Generar ejemplos adversarios
        perturbed_batch_x = adv_attack(loss_f=loss_f, batch_x=batch_x)
        
        # Obtener predicciones
        with torch.no_grad():
            outputs = model(perturbed_batch_x)
            _, predicted = torch.max(outputs.data, 1)
        
        # Mostrar ejemplos adversarios
        for i in range(n_examples):
            # Calcular la perturbación
            perturbation = (perturbed_batch_x[i] - batch_x[i]).abs().cpu()
            
            # Mostrar imagen adversaria
            axes[i, j+1].imshow(perturbed_batch_x[i].cpu().squeeze(), cmap='gray')
            axes[i, j+1].set_title(f'α={alpha:.4f}\nPred: {predicted[i].item()}')
            axes[i, j+1].axis('off')
            
            # Mostrar la perturbación en un subplot adicional
            if j == len(alpha_values) - 1:  # Solo para el último alpha
                fig_pert, ax_pert = plt.subplots(1, 1, figsize=(3, 3))
                im = ax_pert.imshow(perturbation.squeeze(), cmap='hot')
                ax_pert.set_title(f'Perturbación (α={alpha:.4f})')
                plt.colorbar(im, ax=ax_pert)
                ax_pert.axis('off')
                plt.show()
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar estadísticas de precisión y magnitud de perturbación
    print("\nEstadísticas:")
    print("α\tPrecisión\tMagnitud media de perturbación")
    print("-" * 50)
    
    for alpha in alpha_values:
        loss_f = partial(loss, 
                        model=model,
                        batch_y=batch_y,
                        alpha1=default_loss_params['alpha1'],
                        alpha2=default_loss_params['alpha2'],
                        objective=default_loss_params['objective'],
                        force_class=default_loss_params['force_class'],
                        change_expl=default_loss_params['change_expl'])
        
        adv_attack = partial(attack, 
                           eps=eps,
                           alpha=alpha,
                           iters=40,
                           random_start=True)
        
        perturbed_batch_x = adv_attack(loss_f=loss_f, batch_x=batch_x)
        
        with torch.no_grad():
            outputs = model(perturbed_batch_x)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == batch_y).sum().item()
            accuracy = 100 * correct / batch_y.size(0)
            
            # Calcular magnitud media de la perturbación
            perturbation = (perturbed_batch_x - batch_x).abs()
            mean_perturbation = perturbation.mean().item()
            
        print(f"{alpha:.4f}\t{accuracy:.2f}%\t\t{mean_perturbation:.6f}")



def adversarial_accuracy_test(model, test_loader, loss, attack):
    """
    Test the adversarial accuracy of the model.
    Args:
        model (torch.nn.Module): The model to test
        test_loader (torch.utils.data.DataLoader): The test loader
        loss (function): The loss function
        attack (function): The attack function
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_ac = 0
    test_ac_adv = 0

    # Initialize counters for evaluation metrics
    correct_clean = 0
    correct_adv = 0
    total = 0

    for i, batch in enumerate(test_loader):
        batch_x = batch[0]
        batch_y = batch[1]
        batch_x = batch_x.to(device)
        batch_x.requires_grad = True
        batch_y = batch_y.to(device)

        # Handle channel conversion if needed
        if batch_x.shape[1] == 3:
            grayscale_images = 0.2989 * batch_x[:,0] + 0.5870 * batch_x[:,1] + 0.1140 * batch_x[:,2]
            grayscale_images = grayscale_images.unsqueeze(1)
        else:
            grayscale_images = batch_x

        pred_y = model.forward(grayscale_images)
        pred_y = softmax(pred_y)

        loss_f = partial(loss, batch_y=batch_y)
        adv_attack = partial(attack, loss_f=loss_f)

        perturbed_batch_x = adv_attack(grayscale_images)

        pred_y_adv = model.forward(perturbed_batch_x)
        pred_y_adv = softmax(pred_y_adv)

        # test accuracy
        conf_y, max_indices = torch.max(pred_y,1)
        n = max_indices.size(0)
        test_ac += (max_indices == batch_y).sum(dtype=torch.float32)/n
        correct_clean += (max_indices == batch_y).sum().item()

        #adversarial test accuracy
        conf_y_adv, max_indices_adv = torch.max(pred_y_adv,1)
        n = max_indices_adv.size(0)
        test_ac_adv += (max_indices_adv == batch_y).sum(dtype=torch.float32)/n
        correct_adv += (max_indices_adv == batch_y).sum().item()
        
        total += batch_y.size(0)

    # Calculate final metrics
    clean_acc = 100 * correct_clean / total
    adv_acc = 100 * correct_adv / total
    attack_success = 100 * (correct_clean - correct_adv) / correct_clean if correct_clean > 0 else 0

    print("\nEvaluation Metrics:")
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    print(f"Adversarial Accuracy: {adv_acc:.2f}%")
    print(f"Attack Success Rate: {attack_success:.2f}%")
    
    return clean_acc, adv_acc, attack_success