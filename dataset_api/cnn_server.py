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

class CNN_Server_kafka:
    def __init__(self, service_ip="127.0.0.1:9092", raw_data_topic_name = b"vector_raw_data", processed_data_topic_name = b"vector_processed_data", model=None, image_size=28, image_height=1):
        self.client = client = KafkaClient(hosts=service_ip)
        client = KafkaClient(hosts=service_ip)
        print(client.topics)
        self.topic_raw = client.topics[raw_data_topic_name]
        self.topic_processed = client.topics[processed_data_topic_name]
        self.raw_data_topic_name = raw_data_topic_name
        self.processed_data_topic_name
        self.image_size = image_size
        self.image_height = image_height

    def start(self, consumer_group="cnn_server"):
        consumer = topic_raw.get_simple_consumer(
            consumer_group=consumer_group,
            auto_offset_reset=OffsetType.EARLIEST,
            reset_offset_on_start=False,
            auto_commit_enable=True,
            auto_commit_interval_ms=1000
        )
        for message in consumer:
            if message is not None:
                print("CNN server: Message received from topic (" +  self.raw_data_topic_name.decode('UTF-8') + ") with offset: " + str(message.offset))
                raw_data = np.frombuffer(message.value, dtype=np.uint8)
                original_data = torch.tensor(raw_data.reshape(1,self.Image_Height,self.Image_Size,self.Image_Size),dtype=torch.float)/255.0
                model_output, _ = model(original_data)
                #reshape to 1, 1, 28, 28
                prediction = torch.max(model_output, 1)[1].item()
                print("CNN server: predicts image of label: "+ str(prediction))
                with topic_processed.get_producer(min_queued_messages=1, max_queued_messages=1, delivery_reports=True) as producer:
                    producer.produce(int(prediction).to_bytes(1,"big"))
                    msg, exc = producer.get_delivery_report(block=True)
                    if exc is not None:
                        print("Failed to deliver msg {}: {}".format(msg.partition_key, repr(exc)))
                    else:
                        print("Successfully delivered msg {}".format(msg.partition_key))
