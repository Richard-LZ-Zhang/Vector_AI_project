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
import traceback

import json
from google.auth import jwt
from google.cloud import pubsub_v1


class CNN_Server:
    def __init__(self, cloud_name, config_file_path, model, image_height, image_size):
        config = json.load(open(config_file_path))
        if cloud_name == "gcloud":
            project_id = config["project_id"]
            auth_key_path = config["auth_key_path"]
            receiver_sub_id = config["gcloud_receiver_sub_id"]
            gcloud_cnn_server_sub_id = config["gcloud_cnn_server_sub_id"]
            gcloud_raw_data_topic_id = config["gcloud_raw_data_topic_id"]
            gcloud_processed_data_topic_id = config["gcloud_processed_data_topic_id"]
            self.service = CNN_Server_Gcloud(
                project_id,
                gcloud_cnn_server_sub_id,
                gcloud_processed_data_topic_id,
                auth_key_path,
                model,
                image_height,
                image_size,
            )
        elif cloud_name == "kafka":
            kafka_service_ip = config["kafka_service_ip"]
            kafka_raw_data_topic_name = config["kafka_raw_data_topic_name"]
            kafka_processed_data_topic_name = config["kafka_processed_data_topic_name"]
            kafka_raw_data_dev_topic_name = config["kafka_raw_data_dev_topic_name"]
            kafka_processed_data_dev_topic_name = config[
                "kafka_processed_data_dev_topic_name"
            ]
            self.service = CNN_Server_Kafka(
                kafka_service_ip,
                kafka_raw_data_topic_name,
                kafka_processed_data_topic_name,
                model,
                image_size,
                image_height,
            )
        else:
            print("Receiver Object corrupted!")


class CNN_Server_Gcloud:
    def __init__(
        self,
        project_id,
        cnn_server_sub_id,
        gcloud_processed_data_topic_id,
        auth_key_path,
        model,
        image_height,
        image_size,
    ):
        self.model = model
        self.image_size = image_size
        self.image_height = image_height

        service_account_info = json.load(open(auth_key_path))
        credentials_sub = jwt.Credentials.from_service_account_info(
            service_account_info,
            audience="https://pubsub.googleapis.com/google.pubsub.v1.Subscriber",
        )
        credentials_pub = jwt.Credentials.from_service_account_info(
            service_account_info,
            audience="https://pubsub.googleapis.com/google.pubsub.v1.Publisher",
        )

        # Initialize a Subscriber client
        self.subscriber_client = pubsub_v1.SubscriberClient(credentials=credentials_sub)
        # Create a fully qualified identifier in the form of
        # projects/{project_id}/subscriptions/{subscription_id}
        self.raw_data_sub_path = self.subscriber_client.subscription_path(
            project_id, cnn_server_sub_id
        )

        self.publisher_client = pubsub_v1.PublisherClient(credentials=credentials_pub)
        self.processed_data_topic_path = self.publisher_client.topic_path(
            project_id, gcloud_processed_data_topic_id
        )
        self.futures_sub = []
        self.futures_pub = []

    def start(self, time_out=200):
        def callback(message: pubsub_v1.subscriber.message.Message) -> None:
            print("CNN Server: received a message from gcloud.")
            # Acknowledge the message. Unack'ed messages will be redelivered.
            message.ack()
            print(f"CNN Server: Acknowledged {message.message_id}.")

            prediction_byte = byte_to_prediction_byte(
                self.model, message.data, self.image_height, self.image_size
            )

            future = self.publisher_client.publish(
                self.processed_data_topic_path, prediction_byte
            )
            self.futures_pub.append(future)
            message_id = future.result()
            print(
                f"CNN Server: Published a data to Gcloud {self.processed_data_topic_path}: {message_id}"
            )

        future = self.subscriber_client.subscribe(
            self.raw_data_sub_path, callback=callback
        )
        self.futures_sub.append(future)
        print(f"CNN server: Listening for messages on {self.raw_data_sub_path}..\n")

    def hold(self, time_out=200):
        try:
            # Calling result() on StreamingPullFuture keeps the main thread from
            # exiting while messages get processed in the callbacks.
            for future in self.futures_pub:
                future.result(timeout=5)  # a small time for publish threads
            for future in self.futures_sub:
                future.result(timeout=time_out)  # a small time for publish threads
        except Exception as exp:  # noqa
            print("CNN Server Gcloud shutdown due to Exception: ")
            print(exp)
            traceback.print_exc()
            future.cancel()  # Trigger the shutdown.
            future.result()  # Block until the shutdown is complete.

    def close_all(self):
        for future in self.futures_sub:
            future.cancel()
        for future in self.futures_pub:
            future.cancel()


class CNN_Server_Kafka:
    def __init__(
        self,
        service_ip="127.0.0.1:9092",
        raw_data_topic_name=b"vector_raw_data",
        processed_data_topic_name=b"vector_processed_data",
        model=None,
        image_size=28,
        image_height=1,
    ):
        self.raw_data_topic_name = raw_data_topic_name
        self.processed_data_topic_name = processed_data_topic_name
        self.image_size = image_size
        self.image_height = image_height

        self.client = KafkaClient(hosts=service_ip)
        # print(self.client.topics)
        self.topic_raw = self.client.topics[raw_data_topic_name]
        self.topic_processed = self.client.topics[processed_data_topic_name]
        self.model = model
        self.consumer = None
        self.producer = None

    def start(self, consumer_group="cnn_server"):
        self.consumer = self.topic_raw.get_simple_consumer(
            consumer_group=consumer_group,
            auto_offset_reset=OffsetType.LATEST,
            reset_offset_on_start=False,
            auto_commit_enable=True,
            auto_commit_interval_ms=1000,
        )
        self.producer = self.topic_processed.get_producer(
                    min_queued_messages=1, max_queued_messages=1, delivery_reports=True
        )
        for message in self.consumer:
            if message is not None:
                print(
                    "CNN server: Message received from topic ("
                    + self.raw_data_topic_name
                    + ") with offset: "
                    + str(message.offset)
                )
                prediction_byte = byte_to_prediction_byte(
                    self.model, message.value, self.image_height, self.image_size
                )
                self.producer.produce(prediction_byte)
                try:
                    msg, exc = self.producer.get_delivery_report(block=False)
                    if exc is not None:
                        print(
                            "CNN Server: Failed to deliver msg {}: {}".format(
                                msg.partition_key, repr(exc)
                            )
                        )
                    else:
                        # print("CNN Server: Successfully delivered msg {}".format(msg.partition_key))
                        print("CNN Server: Successfully delivered msg {}".format(msg.offset))
                except queue.Empty:
                    print("CNN Server: Queue Empty")

    def hold(self):
        pass

    def close_all(self):
        print("CNN_Server: stop consumer thread.")
        self.consumer.stop()
        self.producer.stop()



def byte_to_prediction_byte(model, byte, image_height, image_size):
    # value is in byte, converted to  raw_data in numpy
    raw_data = np.frombuffer(byte, dtype=np.uint8)
    original_data = (
        torch.tensor(
            raw_data.reshape(1, image_height, image_size, image_size), dtype=torch.float
        )
        / 255.0
    )
    model_output, _ = model(original_data)
    # reshape to 1, 1, 28, 28
    prediction = torch.max(model_output, 1)[1].item()
    print("CNN server: predicts image of label: " + str(prediction))
    return int(prediction).to_bytes(1, "big")
