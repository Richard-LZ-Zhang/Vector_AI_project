""" This code tests custom data_set fed by numpy array data, uint8."""
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
Random_Rotation_Angle = 10

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

print("Max element for original train_data, test_data")
print(torch.max(train_data.train_data).item(), torch.max(test_data.test_data).item())

print("Test custom dataset")

np_train_data = train_data.train_data.numpy().astype(np.uint8)
np_train_labels = train_data.train_labels.numpy().astype(np.uint8)


np_test_data = test_data.test_data.numpy().astype(np.uint8)
np_test_labels = test_data.test_labels.numpy().astype(np.uint8)

print("print np data shape. train_data, train_label, test_data, test_label")
print(np_train_data.shape, np_train_labels.shape, np_test_data.shape, np_test_labels.shape)

np_train_dataset = Dataset_customize(
    np_train_data, np_train_labels, transform=augmentation, train=True
)
# np_train_dataset object accepts np array uint 8 as an input

np_test_dataset = Dataset_customize(
    np_test_data, np_test_labels, transform=augmentation, train=False
)

print_image_shape(np_train_dataset) # plot one example of transformed, and one untransfored
print_image_sample(
    np_train_dataset, index=10
)  # note 


print("Max element for np train_data, test_data")
print(torch.max(np_train_dataset.train_data).item(), torch.max(np_test_dataset.test_data).item())


cnn = CNN(Image_Height, Image_Size, Dropout_Universal)
train_loader = Data.DataLoader(dataset=np_train_dataset, batch_size=BATCH_SIZE, shuffle=True)
New_model = Model(train_loader, np_test_dataset, cnn, lr=LR, test_size=TEST_SIZE)

CNN_trained = New_model.train(Epoch=EPOCH)

print("Numpy test successful!")
