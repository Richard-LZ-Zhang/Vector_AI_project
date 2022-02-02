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


class Receiver_Kafka:
    def __init__(self, service_ip="127.0.0.1:9092", raw_data_topic_name = b"vector_raw_data", processed_data_topic_name = b"vector_processed_data", image_size=28, image_height=1):
        self.client =KafkaClient(hosts=service_ip)
        print(self.client.topics)
        self.service_ip = service_ip
        self.raw_data_topic_name = raw_data_topic_name
        self.processed_data_topic_name = processed_data_topic_name
        self.image_size = image_size
        self.image_height = image_height

    def start(self, consumer_group="processed_data_listenner"):
        topic_raw = self.client.topics[self.raw_data_topic_name]
        topic_processed = self.client.topics[self.processed_data_topic_name]

        consumer = topic_processed.get_simple_consumer(
            consumer_group=consumer_group,
            auto_offset_reset=OffsetType.EARLIEST,
            reset_offset_on_start=False,
            auto_commit_enable=True,
            auto_commit_interval_ms=1000
        )
        for message in consumer:
            if message is not None:
                print("Receiver: Message received from topic (" +  self.processed_data_topic_name.decode('UTF-8') + ") with offset: " + str(message.offset))
                value = int.from_bytes(message.value, "big")
                print("Value: ", value)