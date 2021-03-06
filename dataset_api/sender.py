from pykafka import KafkaClient

import json
import queue
import traceback

from google.auth import jwt
from google.cloud import pubsub_v1
from google.cloud.pubsub import types


class Sender:
    def __init__(self, cloud_name, config_file_path):
        config = json.load(open(config_file_path))
        if cloud_name == "gcloud":
            project_id = config["project_id"]
            auth_key_path = config["auth_key_path"]
            receiver_sub_id = config["gcloud_receiver_sub_id"]
            gcloud_cnn_server_sub_id = config["gcloud_cnn_server_sub_id"]
            gcloud_raw_data_topic_id = config["gcloud_raw_data_topic_id"]
            gcloud_processed_data_topic_id = config["gcloud_processed_data_topic_id"]
            self.service = Sender_Gcloud(
                project_id, gcloud_raw_data_topic_id, auth_key_path
            )
        elif cloud_name == "kafka":
            kafka_service_ip = config["kafka_service_ip"]
            kafka_raw_data_topic_name = config["kafka_raw_data_topic_name"]
            kafka_processed_data_topic_name = config["kafka_processed_data_topic_name"]
            kafka_raw_data_dev_topic_name = config["kafka_raw_data_dev_topic_name"]
            kafka_processed_data_dev_topic_name = config[
                "kafka_processed_data_dev_topic_name"
            ]
            self.service = Sender_Kafka(
                kafka_service_ip,
                kafka_raw_data_topic_name,
                kafka_processed_data_topic_name,
            )
        else:
            print("Receiver Object corrupted!")


class Sender_Gcloud:
    def __init__(self, project_id, topic_id, auth_key_path):
        service_account_info = json.load(open(auth_key_path))
        audience = "https://pubsub.googleapis.com/google.pubsub.v1.Publisher"
        credentials = jwt.Credentials.from_service_account_info(
            service_account_info, audience=audience
        )
        # Initialize a publisher client with batch size of max 500
        self.publisher = pubsub_v1.PublisherClient(
            batch_settings=types.BatchSettings(max_messages=500),
            credentials=credentials,
        )
        self.topic_path = self.publisher.topic_path(project_id, topic_id)
        self.futures = []

    def start(self, messages):
        def callback(future):
            message_id = future.result()
            print(f"Sender: Published a data to Gcloud {self.topic_path}: {message_id}")

        for message in messages:
            future = self.publisher.publish(self.topic_path, message)
            future.add_done_callback(callback)
            self.futures.append(future)

    def hold(self, time_out=200):
        print("Sender: Waiting for all futures to end.")
        for future in self.futures:
            try:
                # Calling result() on StreamingPullFuture keeps the main thread from
                # exiting while messages get processed in the callbacks.
                future.result(timeout=time_out)
            except Exception as exp:
                print("Sender Gcloud publisher shutdown due to exception.")
                print(exp)
                traceback.print_exc()
                future.cancel()  # Trigger the shutdown.
                future.result()  # Block until the shutdown is complete.
        print("Sender: All futures ended.")

    def close_all(self):
        for future in self.futures:
            future.cancel()


class Sender_Kafka:
    def __init__(
        self,
        service_ip="127.0.0.1:9092",
        raw_data_topic_name=b"vector_raw_data",
        processed_data_topic_name=b"vector_processed_data",
        image_size=28,
        image_height=1,
    ):
        self.service_ip = service_ip
        self.client = KafkaClient(hosts=service_ip)
        # print(self.client.topics)
        self.raw_data_topic_name = raw_data_topic_name
        self.processed_data_topic_name = processed_data_topic_name
        # self.image_size = image_size
        # self.image_height = image_height
        self.producer = None

    def start(self, messages):
        topic_raw = self.client.topics[self.raw_data_topic_name]
        self.producer = topic_raw.get_producer(
            max_queued_messages=1000,
            min_queued_messages=10,
            linger_ms=50,
            delivery_reports=True,
        )
        for message in messages:
            self.producer.produce(message)

    def hold(self):
        print("waiting for all messages to be written")
        while True:
            try:
                msg, exc = self.producer.get_delivery_report(block=False)
                if exc is not None:
                    print("Sender: Failed to deliver msg {}: {}".format(msg.offset, repr(exc)))
                else:
                    print("Sender: Successfully delivered msg {}".format(msg.offset))
            except queue.Empty:
                print("Sender: Queue empty")
                break
        self.producer._wait_all()
