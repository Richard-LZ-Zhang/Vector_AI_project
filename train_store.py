# library
# standard library
import os

# third-party library
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

# user library
from ml.cnn import labels_map, augmentation_transform, Dataset_customize, CNN, Model
from ml.cnn import print_image_sample, print_image_shape, get_FashionMNIST_data
from dataset_api.cnn_server import CNN_Server_Kafka, CNN_Server_Gcloud, CNN_Server


# Preprocessing Parameters
Random_Crop_Ratio = 1
Random_Rotation_Angle = 0

# Hyper Parameters
EPOCH = 6  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001  # learning rate
Dropout_Universal = 0
DOWNLOAD_MNIST = False
TEST_SIZE = 2000

Image_Size = 28  # TBD
Image_Height = 1

Fasion_minist_dir = "./ml/Fasion_mnist/"
model_dir = "./ml/trained_model/CNN_model1.pt"

augmentation = augmentation_transform(
    Image_Size, Random_Crop_Ratio, Random_Rotation_Angle
)
train_data, test_data = get_FashionMNIST_data(
    root=Fasion_minist_dir, augmentation=augmentation
)


# print_image_shape(train_data)

# # plot one example of transformed, and one untransfored

print_image_sample(
    train_data, index=30
)  # note that train_data is needed, not test_data


cnn = CNN(Image_Height, Image_Size, Dropout_Universal)
print(cnn)  # net architecture

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)

model = Model(train_loader, test_data, cnn, lr=LR, test_size=TEST_SIZE)

CNN_trained = model.train(Epoch=EPOCH)

# print 10 predictions from test data
test_x = (
    torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:TEST_SIZE]
    / 255.0
)  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:TEST_SIZE]

CNN_trained.eval()
test_output, _ = CNN_trained(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, "prediction number")
print(test_y[:10].numpy(), "real number")

save_root = model_dir
torch.save(CNN_trained.state_dict(), save_root)
print("Saving Model to: " + save_root)
