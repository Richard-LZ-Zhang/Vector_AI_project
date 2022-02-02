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

import json
from google.auth import jwt
from google.cloud import pubsub_v1

class Sender_Gcloud:
    def __init__(self,project_id,subscription_id, auth_key_path, timeout=200.0):
        service_account_info = json.load(open(auth_key_path))
        audience = "https://pubsub.googleapis.com/google.pubsub.v1.Subscriber"
        credentials = jwt.Credentials.from_service_account_info(
            service_account_info, audience=audience
        )
        # Initialize a Subscriber client
        self.subscriber_client = pubsub_v1.SubscriberClient(credentials=credentials)
        # Create a fully qualified identifier in the form of
        # projects/{project_id}/subscriptions/{subscription_id}
        self.subscription_path = subscriber_client.subscription_path(project_id, subscription_id)

    def start(self):
        def callback(message: pubsub_v1.subscriber.message.Message) -> None:
            print(f"Received {message}.")
            # Acknowledge the message. Unack'ed messages will be redelivered.
            message.ack()
            print(f"Acknowledged {message.message_id}.")
            future = self.subscriber_client.subscribe(
                self.subscription_path, callback=callback
            )
            print(f"Listening for messages on {subscription_path}..\n")

            try:
                # Calling result() on StreamingPullFuture keeps the main thread from
                # exiting while messages get processed in the callbacks.
                future.result(timeout=timeout)
            except:  # noqa
                future.cancel()  # Trigger the shutdown.
                print("Sender_Gcloud subscriber shutdown due to an Exception")
                future.result()  # Block until the shutdown is complete.
            # self.subscriber_client.close()


class Sender_Kafka:
    def __init__(self, service_ip="127.0.0.1:9092", raw_data_topic_name = b"vector_raw_data", processed_data_topic_name = b"vector_processed_data", image_size=28, image_height=1):
        self.service_ip = service_ip
        self.client =KafkaClient(hosts=service_ip)
        print(self.client.topics)
        self.raw_data_topic_name = raw_data_topic_name
        self.processed_data_topic_name = processed_data_topic_name
        self.image_size = image_size
        self.image_height = image_height

    def start(self, messages):
        topic_raw = self.client.topics[self.raw_data_topic_name]
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