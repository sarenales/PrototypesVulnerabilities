import torch
import torch.nn as nn
import os
import sys
import time
import matplotlib.pyplot as plt
from tqdm import tqdm, tqdm_notebook
from .config import *
from .data import *
from .autoencoders import *
from .utils import *
from .cae_model import *
from .autoencoders import print_and_write
from .loss_functions import *

# train the model
def train_model(model_type, loss_type):
    
    
    start_time= time.time()

    # download MNIST data
    train_loader, val_loader = get_train_val_loader(data_folder, batch_size, random_seed, augment=False, val_size=0.2,
                            shuffle=True, show_sample=False, num_workers=0, pin_memory=True)
    test_loader = get_test_loader(data_folder, batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # model selection
    if model_type == 'S15':
        n_prototypes =  15   
        model = S15(input_shape=input_shape, n_maps=n_maps, n_prototypes=n_prototypes, 
                    n_layers=n_layers, n_classes=n_classes).to(device)
    elif model_type == 'S30':
        n_prototypes =  30
        model = S30(input_shape=input_shape, n_maps=n_maps, n_prototypes=n_prototypes, 
                    n_layers=n_layers, n_classes=n_classes).to(device)
    elif model_type == 'B30':
        n_prototypes =  30
        model = B30(input_shape=input_shape, n_maps=n_maps, prototypes_by_class=3, 
                    n_layers=n_layers, n_classes=n_classes, fill_with=0.5).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    # save configuration clearly
    model_folder = os.path.join(os.getcwd(), "saved_model", "mnist_model", model_type)
    makedirs(model_folder)
    img_folder = os.path.join(model_folder, "img")
    makedirs(img_folder)
    model_filename = "mnist_cae"

    # console_log is the handle to a text file that records the console output
    log_folder=os.path.join(model_folder, "log")
    makedirs(log_folder)
    console_log = open(os.path.join(log_folder, "console_log.txt"), "w+")


    config_file_path = os.path.join(model_folder, "config.txt")
    with open(config_file_path, "w") as config_file:
        config_file.write("Model Configuration\n")
        config_file.write("Model Type: " + model_type + "\n")
        config_file.write("learning_rate: " + str(learning_rate) + "\n")
        config_file.write("training_epochs: " + str(training_epochs) + "\n")
        config_file.write("batch_size: " + str(batch_size) + "\n")
        config_file.write("test_display_step: " + str(test_display_step) + "\n")
        config_file.write("Loss Type: " + loss_type + "\n")
        config_file.write("Lambda Class - E "+ str(lambda_class) + "\n")
        config_file.write("Lambda Encoder - A"+ str(lambda_ae) + "\n")
        config_file.write("Lambda R1"+ str(lambda_1) + "\n")
        config_file.write("Lambda R2"+ str(lambda_2) + "\n")
        config_file.write("Number of Prototypes: " + str(n_prototypes) + "\n")
        config_file.write("Random Seed: " + str(random_seed) + "\n")


    # training loop
    for epoch in range(0, training_epochs):
        print_and_write("#"*80, console_log)
        print_and_write("Epoch: %04d" % (epoch+1)+"/%04d" % (training_epochs), console_log)
        n_train_batch = len(train_loader)
        n_val_batch = len(val_loader)
        n_test_batch = len(test_loader)
        start = time.time()

        train_ce, train_ae, train_e1, train_e2, train_te, train_ac = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        with tqdm(total=len(train_loader), file=sys.stdout) as pbar:
            for i, batch in enumerate(train_loader):
                batch_x = batch[0]
                batch_y = batch[1]

                # store original batch shape to put it back into this shape after transformation
                batch_shape = batch_x.shape

                # apply elastic transform
                elastic_batch_x = batch_elastic_transform(batch_x.view(batch_size, -1), sigma=sigma, alpha=alpha, height=input_height, width=input_width)
                elastic_batch_x = torch.reshape(torch.tensor(elastic_batch_x), batch_shape)
                elastic_batch_x = elastic_batch_x.to(device)

                batch_y = batch_y.to(device)

                optimizer.zero_grad()        

                pred_y = model.forward(elastic_batch_x)


                # default loss function
                if loss_type == 'default':
                    train_te, train_ce, train_ae, train_e1, train_e2, prototype_distances = loss_f_default(model, elastic_batch_x, batch_y, pred_y, lambda_class, lambda_ae, lambda_1, lambda_2)

                train_te.backward()

                optimizer.step()

                # train accuracy
                max_vals, max_indices = torch.max(pred_y,1)
                n = max_indices.size(0)
                train_ac += (max_indices == batch_y).sum(dtype=torch.float32)/n

                pbar.set_description('batch: %03d' % (1 + i))
                pbar.update(1)
        
        train_ac /= n_train_batch
        print_and_write("training set errors:"+"\tclassification error: {:.6f}".format(train_ce)+
                        "\tautoencoder error: {:.6f}".format(train_ae)+
                        "\terror_1: {:.6f}".format(train_e1)+
                        "\terror_2: {:.6f}".format(train_e2)+
                        "\ttotal error: {:.6f}".format(train_te)+
                        "\taccuracy: {:.6f}".format(train_ac), console_log)
        print_and_write('training takes {0:.2f} seconds.'.format((time.time() - start)), console_log)
        


        ########################################################
        # validation set error terms evaluation
        val_ce, val_ae, val_e1, val_e2, val_te, val_ac = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        with tqdm(total=len(val_loader), file=sys.stdout) as pbar:
            for i, batch in enumerate(val_loader):
                batch_x = batch[0]
                batch_y = batch[1]
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                pred_y = model.forward(batch_x)

                # default loss function
                if loss_type == 'default':
                    train_te, train_ce, train_ae, train_e1, train_e2, prototype_distances = loss_f_default(model, elastic_batch_x, batch_y, pred_y, lambda_class, lambda_ae, lambda_1, lambda_2)

                # validation accuracy
                max_vals, max_indices = torch.max(pred_y,1)
                n = max_indices.size(0)
                val_ac += (max_indices == batch_y).sum(dtype=torch.float32)/n

                pbar.set_description('batch: %03d' % (1 + i))
                pbar.update(1)
            
        val_ac /= n_val_batch
        # after every epoch, check the error terms on the entire training set
        print_and_write("validation set errors:"+"\tclassification error: {:.6f}".format(val_ce)+
                        "\tautoencoder error: {:.6f}".format(val_ae)+
                        "\terror_1: {:.6f}".format(val_e1)+
                        "\terror_2: {:.6f}".format(val_e2)+
                        "\ttotal error: {:.6f}".format(val_te)+
                        "\taccuracy: {:.6f}".format(val_ac), console_log)
        

        ########################################################
        # test set accuracy evaluation
        if (epoch+1) % test_display_step == 0 or epoch == training_epochs - 1:
            test_ac = 0

            for i, batch in enumerate(test_loader):
                batch_x = batch[0]
                batch_y = batch[1]
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                pred_y = model.forward(batch_x)

                # test accuracy
                max_vals, max_indices = torch.max(pred_y,1)
                n = max_indices.size(0)
                test_ac += (max_indices == batch_y).sum(dtype=torch.float32)/n

            test_ac /= n_test_batch

            print_and_write("test set:", console_log)
            print_and_write("\taccuracy: {:.4f}".format(test_ac), console_log)

        if (epoch+1) % save_step == 0 or epoch == training_epochs - 1:
            # save model states
            model_state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1}
            torch.save(model_state, os.path.join(model_folder, model_filename+'%05d.pth' % (epoch+1)))

            # save outputs as images
            # decode prototype vectors
            prototype_imgs = model.decoder(prototype_distances.reshape((-1,10,2,2))).detach().cpu()

            # visualize the prototype images
            # print_and_write("Visualizing the prototype images:", console_log)
            n_cols = 5
            n_rows = n_prototypes // n_cols + 1 if n_prototypes % n_cols != 0 else n_prototypes // n_cols
            g, b = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
            for i in range(n_rows):
                for j in range(n_cols):
                    if i*n_cols + j < n_prototypes:
                        b[i][j].imshow(prototype_imgs[i*n_cols + j].reshape(input_height, input_width),
                                        cmap='gray',
                                        interpolation='none')
                        b[i][j].axis('off')
                        
            plt.savefig(os.path.join(img_folder, 'prototype_result-' + str(epoch+1) + '.png'),
                        transparent=True, bbox_inches='tight', pad_inches=0)
            # plt.show()
            # print_and_write("Visualizing encoded and decoded images:", console_log)

            plt.close()

            # apply encoding and decoding over a small subset of the training set
            batch_x = []
            for batch in train_loader:
                batch_x = batch[0].to(device)
                break

            examples_to_show = 10
            
            encoded = model.encoder.forward(batch_x[:examples_to_show])
            decoded = model.decoder.forward(encoded)

            decoded = decoded.detach().cpu()
            imgs = batch_x.detach().cpu()

            # compare original images to their reconstructions
            f, a = plt.subplots(2, examples_to_show, figsize=(examples_to_show, 2))
            for i in range(examples_to_show):
                a[0][i].imshow(imgs[i].reshape(input_height, input_width),
                                cmap='gray',
                                interpolation='none')
                a[0][i].axis('off')
                a[1][i].imshow(decoded[i].reshape(input_height, input_width), 
                                cmap='gray',
                                interpolation='none')
                a[1][i].axis('off')
                
            plt.savefig(os.path.join(img_folder, 'decoding_result-' + str(epoch+1) + '.png'),
                        transparent=True,
                        bbox_inches='tight',
                        pad_inches=0)
            # plt.show()
            plt.close()
            
    print_and_write('Total taken time {0:.2f} seconds.'.format((time.time() - start_time)), console_log)
    print_and_write("Optimization Finished!", console_log)
    console_log.close()