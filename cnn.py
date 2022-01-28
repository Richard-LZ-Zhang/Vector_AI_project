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

# torch.manual_seed(1)    # reproducible

# Preprocessing Parameters
Random_Crop_Ratio = 1
Random_Rotation_Angle = 0

Image_Size = 28  # TBD
Image_Height = 1

# Hyper Parameters
EPOCH = 10  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 50
LR = 0.001  # learning rate
Dropout_Universal = 0
DOWNLOAD_MNIST = False
TEST_SIZE = 2000

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


# Mnist digits dataset
if not (os.path.exists("./Fasion_mnist/")) or not os.listdir("./Fasion_mnist/"):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True

augmentation_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.RandomCrop(int(Image_Size * Random_Crop_Ratio)),
        transforms.Resize(Image_Size),
        transforms.RandomRotation(Random_Rotation_Angle),
    ]
)  # Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]

train_data = torchvision.datasets.FashionMNIST(
    root="./Fasion_mnist/",
    train=True,  # this is training data
    # transform=torchvision.transforms.ToTensor(),
    transform=augmentation_transform,
    download=DOWNLOAD_MNIST,
)


print("Show train dataset size for data and label")
print(train_data.train_data.size())  # (60000, 28, 28)
print(train_data.train_labels.size())  # (60000)

# plot one example of transformed, and one untransfored

# index = 20
# plt.imshow(train_data[index][0].numpy().reshape(28,28), cmap='gray')
# plt.title('%s' % labels_map[train_data.train_labels[0].item()])
# plt.show()

# plt.imshow(train_data.train_data[index,:,:].numpy().reshape(28,28), cmap='gray')
# plt.title('%s' % labels_map[train_data.train_labels[0].item()])
# plt.show()

# Data Loader for easy mini-batch return in training, the image batch shape will be (50, 1, 28, 28)


# pick a number of samples to speed up testing
test_data = torchvision.datasets.FashionMNIST(root="./Fasion_mnist/", train=False)
test_x = (
    torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor)[:TEST_SIZE]
    / 255.0
)  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
test_y = test_data.test_labels[:TEST_SIZE]

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
    def __init__(self, image_height=Image_Height, dropout_rate=Dropout_Universal):
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
            cnn.parameters(), lr=lr
        )  # optimize all cnn parameters
        self.loss_func = nn.CrossEntropyLoss()

    def train(self, Epoch):
        # training and testing
        for epoch in range(Epoch):
            for step, (b_x, b_y) in enumerate(
                self.train_loader
            ):  # gives batch data, normalize x when iterate train_loader
                output = cnn(b_x)[0]  # cnn output
                loss = self.loss_func(output, b_y)  # cross entropy loss
                self.optimizer.zero_grad()  # clear gradients for this training step
                loss.backward()  # backpropagation, compute gradients
                self.optimizer.step()  # apply gradients

                if step % 500 == 0:
                    test_output, last_layer = cnn(self.test_x)
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
                #     if HAS_SK:
                #         # Visualization of trained flatten layer (T-SNE)
                #         tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                #         plot_only = 500
                #         low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                #         labels = test_y.numpy()[:plot_only]
                # plot_with_labels(low_dim_embs, labels)
        return self.CNN


cnn = CNN()
print(cnn)  # net architecture

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)


# following function (plot_with_labels) is for visualization, can be ignored if not interested
# from matplotlib import cm
# try: from sklearn.manifold import TSNE; HAS_SK = True
# except: HAS_SK = False; print('Please install sklearn for layer visualization')
# def plot_with_labels(lowDWeights, labels):
#     plt.cla()
#     X, Y = lowDWeights[:, 0], lowDWeights[:, 1]
#     for x, y, s in zip(X, Y, labels):
#         c = cm.rainbow(int(255 * s / 9)); plt.text(x, y, s, backgroundcolor=c, fontsize=9)
#     plt.xlim(X.min(), X.max()); plt.ylim(Y.min(), Y.max()); plt.title('Visualize last layer'); plt.show(); plt.pause(0.01)

# plt.ion()


model = Model(train_loader, test_data, CNN, lr=LR,test_size=TEST_SIZE)

# CNN_trained = model.train(Epoch=1)

# # plt.ioff()

# # print 10 predictions from test data
# test_output, _ = cnn(test_x[:10])
# pred_y = torch.max(test_output, 1)[1].data.numpy()
# print(pred_y, "prediction number")
# print(test_y[:10].numpy(), "real number")


print("Test custom dataset")

np_train_data = train_data.train_data.numpy().astype(np.uint8)
np_train_labels = train_data.train_labels.numpy().astype(np.uint8)

np_test_data = test_data.test_data.numpy().astype(np.uint8)
np_test_labels = test_data.test_labels.numpy().astype(np.uint8)

my_dataset = Dataset_customize(np_train_data, np_train_labels, transform=augmentation_transform, train=True)

my_test_data = Dataset_customize(np_test_data, np_test_labels, transform=augmentation_transform, train=False)

cnn = CNN()
train_loader = Data.DataLoader(dataset=my_dataset, batch_size=BATCH_SIZE, shuffle=True)
New_model = Model(train_loader, my_test_data, CNN, lr=LR,test_size=TEST_SIZE)

CNN_trained = New_model.train(Epoch=10)