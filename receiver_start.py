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

from dataset_api.receiver import Receiver_Kafka

Image_Size = 28  # TBD
Image_Height = 1

kafka_service_ip = "127.0.0.1:9092" # "127.0.0.1:9092" # "172.24.217.34:9092"
kafka_raw_data_topic_name = b"vector_raw_data"
kafka_processed_data_topic_name = b"vector_processed_data"
kafka_raw_data_dev_topic_name = b"vector_raw_data_dev"
kafka_processed_data_dev_topic_name = b"vector_processed_data_dev"

receiver_kafka = Receiver_Kafka(
    service_ip=kafka_service_ip,
    raw_data_topic_name=kafka_raw_data_topic_name,
    processed_data_topic_name=kafka_processed_data_topic_name,
    image_size=Image_Size,
    image_height=Image_Height,
)
receiver_kafka.start()

