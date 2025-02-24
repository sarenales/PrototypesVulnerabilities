import torch
import os
import sys
import time
def makedirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




model_folder = os.path.join(os.getcwd(), "saved_model", "mnist_model", "mnist_cae_1")
# makedirs(model_folder)
img_folder = os.path.join(model_folder, "img")
# makedirs(img_folder)
model_filename = "mnist_cae"

# console_log is the handle to a text file that records the console output
log_folder=os.path.join(model_folder, "log")
# makedirs(log_folder)

# Asegurar que la carpeta del modelo y la carpeta de logs existen
# if not os.path.exists(log_folder):
#     os.makedirs(log_folder)  # Crea la carpeta de logs si no existe

# log_path = os.path.join(log_folder, "console_log.txt")

# Asegurar que el archivo existe antes de abrirlo en modo escritura
# if not os.path.exists(log_path):
#    open(log_path, "w").close()  # Crea un archivo vacío si no existe

# console_log = open(log_path, "w+")


# Verificar si la carpeta de logs existe
