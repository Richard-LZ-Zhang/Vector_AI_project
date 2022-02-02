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

# user library
from ml.cnn import labels_map, augmentation_transform, Dataset_customize, CNN, Model
from ml.cnn import print_image_sample, print_image_shape, get_FashionMNIST_data
from dataset_api.sender import Sender_Kafka, Sender_Gcloud

Image_Size = 28  # TBD
Image_Height = 1

kafka_service_ip = "127.0.0.1:9092" # "127.0.0.1:9092" # "172.24.217.34:9092"
kafka_raw_data_topic_name = b"vector_raw_data"
kafka_processed_data_topic_name = b"vector_processed_data"
kafka_raw_data_dev_topic_name = b"vector_raw_data_dev"
kafka_processed_data_dev_topic_name = b"vector_processed_data_dev"


auth_key_path = 'gcloud_key/key.json'
project_id = "vector-project-340115"
gcloud_raw_data_topic_id, gcloud_processed_data_topic_id = "vector-raw-data","vector-processed-data"


Fasion_minist_dir = "./ml/Fasion_mnist/"
model_dir = "./ml/trained_model/CNN_model1.pt"

train_data, test_data = get_FashionMNIST_data(root=Fasion_minist_dir)

#build a list of samples
samples = []
for i in range(1000):
    index = random.randrange(1, 5000, 1) # a random integer from 1 - 5000
    sample_label = test_data.test_labels[index].item()
    print("Sending image in MNIST dataset with label " + str(sample_label))
    sample = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[index]
    # shape from (2000, 28, 28) to (2000, 1, 28, 28), and then pick the image with the index
    sample = np.array(sample.numpy().flatten(),dtype=np.uint8).tobytes()
    # then convert to numpy and flatten and convert to uint8, and to bytes
    samples.append(sample)

sender_gcloud = Sender_Gcloud(project_id, gcloud_raw_data_topic_id, auth_key_path)
sender_gcloud.start(samples)
# sender_kafka = Sender_Kafka(
#     service_ip=kafka_service_ip,
#     raw_data_topic_name=kafka_raw_data_topic_name,
#     processed_data_topic_name=kafka_processed_data_topic_name,
#     image_size=Image_Size,
#     image_height=Image_Height,
# )
# sender_kafka.start(samples)