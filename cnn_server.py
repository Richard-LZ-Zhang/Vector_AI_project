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

model = CNN(Image_Height, Image_Size)
model.load_state_dict(torch.load(model_path))
model.eval()
# These are using samples
# train_data, test_data = get_FashionMNIST_data(root="./Fasion_mnist/")
# # print 10 predictions from test data
# sample = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:10] / 255.0  
# # shape from (2000, 28, 28) to (2000, 1, 28, 28), and then normalize the value in range(0,1)
# sample_label = test_data.test_labels[:10]
# model_output, _ = model(sample)
# prediction = torch.max(model_output, 1)[1].data.numpy()
# print(prediction, "prediction number")
# print(sample_label.numpy(), "real number")

consumer = topic_raw.get_simple_consumer(
    consumer_group="cnn_server",
    auto_offset_reset=OffsetType.EARLIEST,
    reset_offset_on_start=False,
    auto_commit_enable=True,
    auto_commit_interval_ms=1000
)



for message in consumer:
    if message is not None:
        print("CNN server: Message received from topic (" +  raw_data_topic_name.decode('UTF-8') + ") with offset: " + str(message.offset))
        raw_data = np.frombuffer(message.value, dtype=np.uint8)
        original_data = torch.tensor(raw_data.reshape(1,Image_Height,Image_Size,Image_Size),dtype=torch.float)/255.0
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
