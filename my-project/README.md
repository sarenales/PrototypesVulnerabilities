my_project/
├── src/  # Código modularizado
│   ├── config.py  # Parámetros del modelo
│   ├── utils.py  # Funciones auxiliares (GPU setup, etc.)
│   ├── data.py  # Carga y preprocesamiento de datos
│   ├── autoencoders.py  # Módulos de red y helpers
│   ├── cae_model.py  # Definición del modelo CAE
│   ├── training.py  # Entrenamiento del modelo
│   ├── adversarial.py  # Implementación de ataques adversarios
│
├── main.py  # Script para ejecutar experimentos sin notebook
├── requirements.txt  # Librerías necesarias
├── README.md  # Explicación del proyecto
├── attacks_tests.ipynb  # Notebook para evaluar ataques adversarios
├── analisis_train.ipynb  # Notebook principal para ejecutar experimentos

