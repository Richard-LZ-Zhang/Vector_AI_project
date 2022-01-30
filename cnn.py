# library
# standard library
import os

# third-party library
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

from torchvision import transforms

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

def augmentation_transform(image_size, random_crop_ratio,random_rotation_angle):
    return  transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomCrop(int(image_size * random_crop_ratio)),
        transforms.Resize(image_size),
        transforms.RandomRotation(random_rotation_angle),
    ]
)  # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]

class Dataset_customize(Data.Dataset):
    def __init__(self, data, labels, transform=None, train=True):
        super(Dataset_customize,self).__init__()
        self.data = data
        self.len = self.data.shape[0]
        self.labels = labels
        self.transform = transform
        if train == False:
            self.test_data = torch.tensor(data)
            self.test_labels = torch.tensor(labels)
        else:
            self.train_data = torch.tensor(data)
            self.train_labels = torch.tensor(labels)
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        if self.transform:
            x = self.transform(x)
        return x,y

class CNN(nn.Module):
    def __init__(self, image_height=1, image_size=28, dropout_rate=0):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=image_height,  # input height
                out_channels=16,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 28, 28)
            nn.Dropout2d(p=dropout_rate),
            nn.ReLU(),  # activation
            nn.MaxPool2d(
                kernel_size=2
            ),  # choose max value in 2x2 area, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(16, 32, 5, 1, 2),  # output shape (32, 14, 14)
            nn.Dropout2d(p=dropout_rate),
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 7, 7)
        )
        self.out = nn.Linear(32 * 7 * 7, 10)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(
            x.size(0), -1
        )  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x  # return x for visualization


class Model:
    def __init__(self, train_loader, test_data, CNN, lr=0.001, test_size=2000):
        self.test_x = (
            torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[
                :test_size
            ]
            / 255.0
        )  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
        self.test_y = test_data.test_labels[:test_size]
        self.train_loader = train_loader
        self.CNN = CNN
        self.optimizer = torch.optim.Adam(
            self.CNN.parameters(), lr=lr
        )  # optimize all cnn parameters
        self.loss_func = nn.CrossEntropyLoss()

    def train(self, Epoch):
        # training and testing
        for epoch in range(Epoch):
            for step, (b_x, b_y) in enumerate(
                self.train_loader
            ):  # gives batch data, normalize x when iterate train_loader
                output = self.CNN(b_x)[0]  # cnn output
                loss = self.loss_func(output, b_y)  # cross entropy loss
                self.optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                self.optimizer.step()  # apply gradients

                if step % 500 == 0:
                    self.CNN.eval()
                    test_output, last_layer = self.CNN(self.test_x)
                    pred_y = torch.max(test_output, 1)[1].data.numpy()
                    accuracy = float(
                        (pred_y == self.test_y.data.numpy()).astype(int).sum()
                    ) / float(self.test_y.size(0))
                    print(
                        "Epoch: ",
                        epoch,
                        "| train loss: %.4f" % loss.data.numpy(),
                        "| test accuracy: %.2f" % accuracy,
                    )
                    self.CNN.train(True)
                #     if HAS_SK:
                #         # Visualization of trained flatten layer (T-SNE)
                #         tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                #         plot_only = 500
                #         low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                #         labels = test_y.numpy()[:plot_only]
                # plot_with_labels(low_dim_embs, labels)
        self.CNN.train(True)
        return self.CNN


def get_FashionMNIST_data(root="./Fasion_mnist/",augmentation=None,download_MNIST=False):
    if not (os.path.exists(root)) or not os.listdir(root):
    # not mnist dir or mnist is empyt dir
        download_MNIST = True
    train_data = torchvision.datasets.FashionMNIST(
        root=root,
        train=True,  # this is training data
        # transform=torchvision.transforms.ToTensor(),
        transform=augmentation,
        download=download_MNIST,
    )
    test_data = torchvision.datasets.FashionMNIST(root=root, train=False)
    return train_data, test_data


def print_image_sample(train_dataset, index=0):
    # plot one example of transformed, and one untransfored
    index = 20
    plt.imshow(train_dataset[index][0].numpy().reshape(28,28), cmap='gray')
    plt.title('%s' % labels_map[train_dataset.train_labels[0].item()])
    plt.show()

    plt.imshow(train_dataset.train_data[index,:,:].numpy().reshape(28,28), cmap='gray')
    plt.title('%s' % labels_map[train_dataset.train_labels[0].item()])
    plt.show()

def print_image_shape(train_dataset):
    print("Show train dataset size for data untransformed, data trasnformed, and label")
    print(train_dataset.train_data.size())  # (60000, 28, 28)
    print(train_dataset.train_labels.size())  # (60000)
    print(train_dataset[0][0].size()) # (60000, 28, 28)