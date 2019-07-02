import torch
import torch.nn as nn

# Change detection network models
# Assumes 15x15 patches 
# Rodrigo Caye Daudt
# https://rcdaudt.github.io/

class SiamNet_15(nn.Module):
    def __init__(self, n_in = 3):
        super(SiamNet_15, self).__init__()

        self.layer_depth = [n_in, 64, 64, 128, 128, 64, 2]

        self.cnn = nn.Sequential(
            nn.Conv2d(self.layer_depth[0], self.layer_depth[1], kernel_size=3), # n=13
            nn.BatchNorm2d(self.layer_depth[1]), # n=13
            nn.ReLU(), # n=13
            nn.Dropout2d(p=0.2), # n=13
            nn.Conv2d(self.layer_depth[1], self.layer_depth[2], kernel_size=3), # n=11
            nn.BatchNorm2d(self.layer_depth[2]), # n=11
            nn.ReLU(), # n=11
            nn.Dropout2d(p=0.2), # n=11
            nn.Conv2d(self.layer_depth[2], self.layer_depth[3], kernel_size=3), # n=9
            nn.BatchNorm2d(self.layer_depth[3]), # n=9
            nn.ReLU(), # n=3
            nn.Dropout2d(p=0.2), # n=9
            nn.Conv2d(self.layer_depth[3], self.layer_depth[4], kernel_size=3), # n=7
            nn.BatchNorm2d(self.layer_depth[4]), # n=7
            nn.ReLU() # n=7
            )

        self.fc = nn.Sequential(
            nn.Linear(2*7*7*self.layer_depth[4], self.layer_depth[5]),
            nn.BatchNorm2d(self.layer_depth[5]), 
            nn.ReLU(),
            nn.Dropout2d(p=0.2), 
            nn.Linear(self.layer_depth[5], self.layer_depth[6]),
            nn.Softmax()	
            )

    def forward(self, x1, x2):
        output = torch.cat((self.cnn(x1), self.cnn(x2)), 1)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output
