import torch
import torch.nn as nn

class linear_3(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(linear_3, self).__init__()
        self.fc1 = nn.Linear(input_channel, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_channel)
        self.softmax = nn.Softmax(-1)

        self.drop = nn.Dropout()
        self.relu = nn.ReLU()

    def forward(self, input_data):
        x1 = self.fc1(input_data)
        x1 = self.relu(x1)
        x1 = self.drop(x1)
        x2 = self.fc2(x1)
        x2 = self.relu(x2)
        x2 = self.drop(x2)
        x3 = self.fc3(x2)
        x3 = self.softmax(x3)
        return x1, x2, x3