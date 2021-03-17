import torch
import torchvision
import cv2

import numpy as np 
import pandas as pd
import torch.nn as nn
import torch.utils.data as utils
import matplotlib.pyplot as plt


from sklearn.preprocessing import LabelBinarizer
from PIL import Image


class SignMnistDataset(utils.Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        mnist_data = pd.read_csv(root_dir + "/" + csv_file)
        self.labels = torch.LongTensor(mnist_data['label'].values.tolist())
        self.image_data = mnist_data.drop('label', axis=1).values
        newdata = []
        for i in self.image_data:
            newdata.append(i.reshape(1, 28, 28))
        
        self.image_data = torch.FloatTensor(np.array(newdata))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_data = self.image_data[idx]
        label = self.labels[idx]
        
        sample = {'image': img_data, 'label': label}
        return sample

class E2EModel(nn.Module):
    def __init__(self):
        super(E2EModel, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 24, 3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(24, 48, 3)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(48, 64, 3)
        self.dropout1 = nn.Dropout2d(0.6)

        self.fc1 = nn.Linear(64 * 3 * 3, 576)
        self.fc2 = nn.Linear(576, 26)

        self.softmax = nn.LogSoftmax(1)

    
    def forward(self, x):
        x = self.conv1(x)
        x = torch.functional.F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = torch.functional.F.relu(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = torch.functional.F.relu(x)
        x = self.dropout1(x)
        x = x.view(-1, 64 * 3 * 3)
        x = torch.functional.F.relu(self.fc1(x))
        x = torch.functional.F.relu(self.fc2(x))

        return self.softmax(x)

def imshow(img):
    npimg = img.numpy()
    # npimg = npimg.reshape(-1, 28, 28)
    plt.imshow(np.transpose(npimg), cmap="gray")
    plt.show()

def accuracy(net, dataloader):
  correct = 0
  total = 0
  with torch.no_grad():
      for batch in dataloader:
          images, labels = batch['image'], batch['label']
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  return correct/total

def main():

    trainset = SignMnistDataset('sign_mnist_train.csv', './data')
    testset = SignMnistDataset('sign_mnist_test.csv', './data')

    trainloader = utils.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)
    testloader = utils.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    model = E2EModel()

    losses = []
    lossfunc = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.7)

    epochs = 30
    for epoch in range(epochs):
        sum_loss = 0.0
        for i, batch in enumerate(trainloader, 0):
            images, labels = batch['image'], batch['label']

            optimizer.zero_grad()

            output = model(images)
            loss = lossfunc(output, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            sum_loss += loss.item()

            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
    
    # plt.plot(losses)
    # plt.show()

    print("Training accuracy: %f" % accuracy(model, trainloader))
    print("Testing accuracy: %f" % accuracy(model, testloader))


    # camera = cv2.VideoCapture(0)
    
    # lettermap = { 0 : 'a', 1 : 'b', 2 : 'c' , 3 : 'd', 4 : 'e', 5 : 'f',
    #              6 : 'g', 7 : 'h', 8 : 'i', 9 : 'j', 10 : 'k', 11 : 'l', 12 : 'm',
    #              13 : 'n', 14 : 'o', 15 : 'p', 16 : 'q', 17 : 'r', 18 : 's', 19 : 't',
    #              20 : 'u', 21 : 'v', 22 : 'w', 23 : 'x', 24 : 'y', 25 : 'z'}

    # while True:
    #     ret, frame = camera.read()

    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     # print(gray.shape)
    #     frame = cv2.resize(gray, (int(gray.shape[1] * 5 / 100), int(gray.shape[0] * 5 / 100)))
    #     # print(frame.shape)
    #     npframe = cv2.resize(frame, (28, 28))
    #     npframe = cv2.transpose(npframe)
    #     npframe = cv2.flip(npframe, flipCode=0)
    #     cv2.imshow('frame2', npframe)
    #     npframe = npframe.reshape(1, 28, 28)

    #     pred = model(torch.FloatTensor(np.array([npframe])))
    #     _, pred = torch.max(pred.data, 1)
    #     print(lettermap[pred.item()])

    #     cv2.putText(gray,  
    #             lettermap[pred.item()],  
    #             (50, 50),  
    #             cv2.FONT_HERSHEY_SIMPLEX, 1,  
    #             (0, 255, 255),  
    #             2,  
    #             cv2.LINE_4)
    #     cv2.imshow('frame', gray)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    
    # camera.release()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    main()