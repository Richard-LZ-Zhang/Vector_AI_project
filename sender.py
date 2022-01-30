from pykafka import KafkaClient
from pykafka.utils.compat import Empty
from pykafka.common import OffsetType

# ML library
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
import os
import random

from cnn import CNN, get_FashionMNIST_data

Image_Size = 28  # TBD
Image_Height = 1

service_ip = "127.0.0.1:9092" # "127.0.0.1:9092" # "172.24.217.34:9092"
raw_data_topic_name = b"vector_raw_data"
processed_data_topic_name = b"vector_processed_data"
raw_data_dev_topic_name = b"vector_raw_data_dev"
processed_data_dev_topic_name = b"vector_processed_data_dev"

client = KafkaClient(hosts=service_ip)
print(client.topics)
topic_raw = client.topics[raw_data_topic_name]

train_data, test_data = get_FashionMNIST_data(root="./Fasion_mnist/")

index = random.randrange(1, 5000, 1) # a random integer from 1 - 5000
sample_label = test_data.test_labels[index].item()
print("Sending image in MNIST dataset with label " + str(sample_label))
sample = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[index]
# shape from (2000, 28, 28) to (2000, 1, 28, 28), and then pick the image with the index
sample = np.array(sample.numpy().flatten(),dtype=np.uint8).tobytes()
# then convert to numpy and flatten and convert to uint8, and to bytes

with topic_raw.get_producer(
    min_queued_messages=1, max_queued_messages=1, delivery_reports=True
) as producer:
    producer.produce(sample)
    msg, exc = producer.get_delivery_report(block=True)
    if exc is not None:
        print("Failed to deliver msg {}: {}".format(msg.partition_key, repr(exc)))
    else:
        print("Successfully delivered a message")

    print("waiting for all messages to be written")
    producer._wait_all()