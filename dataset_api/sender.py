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

class Sender_Kafka:
    def __init__(self, service_ip="127.0.0.1:9092", raw_data_topic_name = b"vector_raw_data", processed_data_topic_name = b"vector_processed_data", image_size=28, image_height=1):
        self.service_ip = service_ip
        self.raw_data_topic_name = raw_data_topic_name
        self.processed_data_topic_name = processed_data_topic_name
        self.image_size = image_size
        self.image_height = image_height

    def start(self, messages):
        client = KafkaClient(hosts=self.service_ip)
        topic_raw = client.topics[self.raw_data_topic_name]
        with topic_raw.get_producer(
            min_queued_messages=1, max_queued_messages=1, delivery_reports=True
        ) as producer:
            for message in messages:
                producer.produce(message)
                msg, exc = producer.get_delivery_report(block=True)
                if exc is not None:
                    print("Failed to deliver msg {}: {}".format(msg.partition_key, repr(exc)))
                else:
                    print("Successfully delivered a message")

            print("waiting for all messages to be written")
            producer._wait_all()