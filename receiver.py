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

from cnn import CNN, get_FashionMNIST_data

Image_Size = 28  # TBD
Image_Height = 1

service_ip = "127.0.0.1:9092" # "127.0.0.1:9092" # "172.24.217.34:9092"
raw_data_topic_name = b"vector_raw_data"
processed_data_topic_name = b"vector_processed_data"
raw_data_dev_topic_name = b"vector_raw_data_dev"
processed_data_dev_topic_name = b"vector_processed_data_dev"

model_path = "./trained_model/CNN_model1.pt"

client = KafkaClient(hosts=service_ip)

print(client.topics)
topic_raw = client.topics[raw_data_topic_name]
topic_processed = client.topics[processed_data_topic_name]

consumer = topic_processed.get_simple_consumer(
    consumer_group="processed_data_listenner",
    auto_offset_reset=OffsetType.EARLIEST,
    reset_offset_on_start=False,
    auto_commit_enable=True,
    auto_commit_interval_ms=1000
)



for message in consumer:
    if message is not None:
        print("Receiver: Message received from topic (" +  processed_data_topic_name.decode('UTF-8') + ") with offset: " + str(message.offset))
        value = int.from_bytes(message.value, "big")
        print("Value: ", value)