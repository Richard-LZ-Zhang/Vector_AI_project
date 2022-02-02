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
from google.auth import jwt
from google.cloud import pubsub_v1



class CNN_Server_Gcloud:
    def __init__(self,project_id,raw_data_topic_name, processed_data_topic_name, auth_key_path, timeout=200.0):
        service_account_info = json.load(open(auth_key_path))
        audience = "https://pubsub.googleapis.com/google.pubsub.v1.Subscriber"
        credentials = jwt.Credentials.from_service_account_info(
            service_account_info, audience=audience
        )
        # Initialize a Subscriber client
        self.subscriber_client = pubsub_v1.SubscriberClient(credentials=credentials)
        # Create a fully qualified identifier in the form of
        # projects/{project_id}/subscriptions/{subscription_id}
        self.raw_data_path = self.subscriber_client.subscription_path(project_id, raw_data_topic_name)

    def start(self):
        def callback(message: pubsub_v1.subscriber.message.Message) -> None:
            print("CNN Server: received a message from gcloud.")
            # Acknowledge the message. Unack'ed messages will be redelivered.
            message.ack()
            print(f"Acknowledged {message.message_id}.")
        future = self.subscriber_client.subscribe(
            self.raw_data_path, callback=callback
        )
        print(f"Listening for messages on {self.raw_data_path}..\n")
        try:
            # Calling result() on StreamingPullFuture keeps the main thread from
            # exiting while messages get processed in the callbacks.
            future.result(timeout=200)
        except:  # noqa
            future.cancel()  # Trigger the shutdown.
            print("Sender_Gcloud subscriber shutdown due to an Exception")
            future.result()  # Block until the shutdown is complete.
        self.subscriber_client.close()

        

class CNN_Server_Kafka:
    def __init__(self, service_ip="127.0.0.1:9092", raw_data_topic_name = b"vector_raw_data", processed_data_topic_name = b"vector_processed_data", model=None, image_size=28, image_height=1):
        self.raw_data_topic_name = raw_data_topic_name
        self.processed_data_topic_name = processed_data_topic_name
        self.image_size = image_size
        self.image_height = image_height

        self.client =KafkaClient(hosts=service_ip)
        print(self.client.topics)
        self.topic_raw = self.client.topics[raw_data_topic_name]
        self.topic_processed = self.client.topics[processed_data_topic_name]
        self.model = model

    def start(self, consumer_group="cnn_server"):
        consumer = self.topic_raw.get_simple_consumer(
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
                original_data = torch.tensor(raw_data.reshape(1,self.image_height,self.image_size,self.image_size),dtype=torch.float)/255.0
                model_output, _ = self.model(original_data)
                #reshape to 1, 1, 28, 28
                prediction = torch.max(model_output, 1)[1].item()
                print("CNN server: predicts image of label: "+ str(prediction))
                with self.topic_processed.get_producer(min_queued_messages=1, max_queued_messages=1, delivery_reports=True) as producer:
                    producer.produce(int(prediction).to_bytes(1,"big"))
                    msg, exc = producer.get_delivery_report(block=True)
                    if exc is not None:
                        print("Failed to deliver msg {}: {}".format(msg.partition_key, repr(exc)))
                    else:
                        print("Successfully delivered msg {}".format(msg.partition_key))
