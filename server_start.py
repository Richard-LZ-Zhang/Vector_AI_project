# ML library
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os

from ml.cnn import CNN, get_FashionMNIST_data
from dataset_api.cnn_server import CNN_Server_Kafka, CNN_Server_Gcloud, CNN_Server

model_path = "ml/trained_model/CNN_model1.pt"

Image_Size = 28  # TBD
Image_Height = 1

model = CNN(Image_Height, Image_Size)
model.load_state_dict(torch.load(model_path))
model.eval()


# These are using samples
# train_data, test_data = get_FashionMNIST_data(root="./ml/Fasion_mnist/")
# # print 10 predictions from test data
# sample = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:10] / 255.0
# # shape from (2000, 28, 28) to (2000, 1, 28, 28), and then normalize the value in range(0,1)
# sample_label = test_data.test_labels[:10]
# model_output, _ = model(sample)
# prediction = torch.max(model_output, 1)[1].data.numpy()
# print(prediction, "prediction number")
# print(sample_label.numpy(), "real number")

# cnn_server = CNN_Server("gcloud", "service_config/gcloud_config.json", model, Image_Height, Image_Size)
cnn_server = CNN_Server(
    "kafka", "service_config/kafka_config.json", model, Image_Height, Image_Size
)
cnn_server.service.start()
cnn_server.service.hold()

# Kafka parameters
# kafka_service_ip = "127.0.0.1:9092"  # "127.0.0.1:9092" # "172.24.217.34:9092"
# kafka_raw_data_topic_name = b"vector_raw_data"
# kafka_processed_data_topic_name = b"vector_processed_data"
# kafka_raw_data_dev_topic_name = b"vector_raw_data_dev"
# kafka_processed_data_dev_topic_name = b"vector_processed_data_dev"
# gcloud_raw_data_topic_id, gcloud_processed_data_topic_id = "vector-raw-data","vector-processed-data"

# Google cloud paramters
# auth_key_path = 'gcloud_key/key.json'
# project_id = "vector-project-340115"
# gcloud_cnn_server_sub_id = "cnn_server"

# cnn_server_gcloud = CNN_Server_Gcloud(project_id,gcloud_cnn_server_sub_id, gcloud_processed_data_topic_id, auth_key_path, model, Image_Height, Image_Size)

# cnn_server_gcloud.start(time_out=100)


# cnn_server_kafka = CNN_Server_Kafka(
#     service_ip=kafka_service_ip,
#     raw_data_topic_name=kafka_raw_data_topic_name,
#     processed_data_topic_name=kafka_processed_data_topic_name,
#     model=model,
#     image_size=Image_Size,
#     image_height=Image_Height,
# )
# cnn_server_kafka.start()
