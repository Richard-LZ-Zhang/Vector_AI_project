import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

import queue
from datetime import datetime
import uuid
import time
import random
import os
import threading

# user library
from ml.cnn import labels_map, augmentation_transform, Dataset_customize, CNN, Model
from ml.cnn import print_image_sample, print_image_shape, get_FashionMNIST_data, get_samples, test_accuracy
from dataset_api.sender import Sender_Kafka, Sender_Gcloud, Sender
from dataset_api.receiver import Receiver_Gcloud, Receiver_Kafka, Receiver
from dataset_api.cnn_server import CNN_Server_Kafka, CNN_Server_Gcloud, CNN_Server

from concurrent.futures import ThreadPoolExecutor

MODE = "gcloud"
Image_Size = 28  # TBD
Image_Height = 1

Fasion_minist_dir = "./ml/Fasion_mnist/"
model_dir = "./ml/trained_model/CNN_model1.pt"

kafka_config_path = "service_config/kafka_config.json"
gcloud_config_path = "service_config/gcloud_config.json"

train_data, test_data = get_FashionMNIST_data(root=Fasion_minist_dir)

image_num = 200
samples = get_samples(image_num, test_data)

model = CNN(Image_Height, Image_Size)
model.load_state_dict(torch.load(model_dir))
model.eval()

accuracy = test_accuracy(test_data, model)

def sender():
    if MODE == "kafka":
        sender = Sender("kafka", kafka_config_path)
    elif MODE == "gcloud":
        sender = Sender("gcloud", gcloud_config_path)
    sender.service.start(samples)
    sender.service.hold()

def server():
    if MODE == "kafka":
        cnn_server = CNN_Server(
        "kafka", kafka_config_path, model, Image_Height, Image_Size
        )
    elif MODE == "gcloud":
        cnn_server = CNN_Server(
        "gcloud", gcloud_config_path, model, Image_Height, Image_Size
        )
    cnn_server.service.start()
    cnn_server.service.hold()

def receiver():
    if MODE == "kafka":
        receiver = Receiver("kafka", kafka_config_path)
    elif MODE == "gcloud":
        receiver = Receiver("gcloud", gcloud_config_path)
    receiver.service.start()
    receiver.service.hold()

sender_thread = threading.Thread(target=sender)
server_thread = threading.Thread(target=server)
receiver_thread = threading.Thread(target=receiver)

sender_thread.start()
server_thread.start()
receiver_thread.start()

sender_thread.join()
server_thread.join()
receiver_thread.join()

