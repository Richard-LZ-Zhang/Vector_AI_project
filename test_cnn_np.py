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

# Preprocessing Parameters
Random_Crop_Ratio = 1
Random_Rotation_Angle = 0

# Hyper Parameters
EPOCH = 1  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001  # learning rate
Dropout_Universal = 0
DOWNLOAD_MNIST = False
TEST_SIZE = 2000

Image_Size = 28  # TBD
Image_Height = 1

Fasion_minist_dir = "./ml/Fasion_mnist/"

augmentation = augmentation_transform(
    Image_Size, Random_Crop_Ratio, Random_Rotation_Angle
)
train_data, test_data = get_FashionMNIST_data(
    root=Fasion_minist_dir, augmentation=augmentation
)

print_image_shape(train_data)

# plot one example of transformed, and one untransfored
print_image_sample(
    train_data, index=10
)  # note that train_data is needed, not test_data

print("Test custom dataset")

np_train_data = train_data.train_data.numpy().astype(np.uint8)
np_train_labels = train_data.train_labels.numpy().astype(np.uint8)

np_test_data = test_data.test_data.numpy().astype(np.uint8)
np_test_labels = test_data.test_labels.numpy().astype(np.uint8)

my_dataset = Dataset_customize(
    np_train_data, np_train_labels, transform=augmentation, train=True
)

my_test_data = Dataset_customize(
    np_test_data, np_test_labels, transform=augmentation, train=False
)

cnn = CNN(Image_Height, Image_Size, Dropout_Universal)
train_loader = Data.DataLoader(dataset=my_dataset, batch_size=BATCH_SIZE, shuffle=True)
New_model = Model(train_loader, my_test_data, cnn, lr=LR, test_size=TEST_SIZE)

CNN_trained = New_model.train(Epoch=1)

print("Numpy test successful!")
