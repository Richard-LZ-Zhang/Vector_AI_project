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
import json
import traceback

from google.auth import jwt
from google.cloud import pubsub_v1

class Receiver:
    def __init__(self, cloud_name, config_file_path):
        if cloud_name == "gcloud":
            config = json.load(open(config_file_path))
            project_id = config["project_id"]
            auth_key_path = config["auth_key_path"]
            receiver_sub_id = config["gcloud_receiver_sub_id"]
            gcloud_cnn_server_sub_id = config["gcloud_cnn_server_sub_id"]
            gcloud_raw_data_topic_id = config["gcloud_raw_data_topic_id"]
            gcloud_processed_data_topic_id = config["gcloud_processed_data_topic_id"]
            self.service = Receiver_Gcloud(project_id, receiver_sub_id, auth_key_path)
        elif cloud_name == "kafka":
            pass
        else:
            print("Receiver Object corrupted!")



class Receiver_Gcloud:
    def __init__(self,project_id, receiver_sub_id, auth_key_path):
        service_account_info = json.load(open(auth_key_path))
        credentials_sub = jwt.Credentials.from_service_account_info(
            service_account_info, audience="https://pubsub.googleapis.com/google.pubsub.v1.Subscriber"
        )
        self.subscriber_client = pubsub_v1.SubscriberClient(credentials=credentials_sub)
        # Create a fully qualified identifier in the form of
        # projects/{project_id}/subscriptions/{subscription_id}
        self.processed_data_sub_path = self.subscriber_client.subscription_path(project_id, receiver_sub_id)
        self.futures = []

    def start(self):
        def callback(message: pubsub_v1.subscriber.message.Message) -> None:
            value = int.from_bytes(message.data, "big")
            print("Receiver Gcloud: received a message from topic processed data. Prediction: {}".format(value))
            # Acknowledge the message. Unack'ed messages will be redelivered.
            message.ack()
            print(f"Acknowledged {message.message_id}.")
        future = self.subscriber_client.subscribe(
            self.processed_data_sub_path, callback=callback
        )
        self.futures.append(future)
        print(f"Listening for messages on {self.processed_data_sub_path}..\n")

    def hold(self, time_out=200):
        for future in self.futures:
            try:
                # Calling result() on StreamingPullFuture keeps the main thread from
                # exiting while messages get processed in the callbacks.
                future.result(timeout=time_out)
            except Exception as exp:
                print("Receiver Gcloud subscriber shutdown due to exception.")
                print(exp)
                traceback.print_exc()
                future.cancel()  # Trigger the shutdown.
                future.result()  # Block until the shutdown is complete.
    
    def close_all(self):
        for future in self.futures:
            future.cancel()
        self.hold()

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