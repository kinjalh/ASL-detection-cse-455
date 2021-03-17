import torch
import numpy

IMG_SIZE = 784

class MSModel(torch.nn.Module):

    def __init__(self):
        super(MSModel, self).__init__()
        
        self.fc1 = torch.nn.Linear(IMG_SIZE, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 25)

        self.actfn1 = torch.nn.LeakyReLU(0.02)
        self.actfn2 = torch.nn.LeakyReLU()
        self.actfn3 = torch.nn.Softmax(dim=1)

        self.dropout1 = torch.nn.Dropout(0.25)
        self.dropout2 = torch.nn.Dropout(0.25)
    
    def forward(
        self,
        x: torch.tensor
    ) -> torch.tensor:
        x = self.fc1(x)
        x = self.actfn1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.actfn2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.actfn3(x)
        return x
