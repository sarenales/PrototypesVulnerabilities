import torch
import os
import sys
import time


def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_folder = os.path.join(os.getcwd(), "saved_model", "mnist_model", "mnist_cae_1")
makedirs(model_folder)
img_folder = os.path.join(model_folder, "img")
makedirs(img_folder)
model_filename = "mnist_cae"

# console_log is the handle to a text file that records the console output
log_folder=os.path.join(model_folder, "log")
makedirs(log_folder)
console_log = open(os.path.join(log_folder, "console_log.txt"), "w+")