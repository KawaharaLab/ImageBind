import os
import torch

class ForceModelCNN(torch.nn.Module):
    def __init__(self, out_dim=512):
        super(ForceModelCNN, self).__init__()
        # use 2D conv treating input as [1 x in_channels x in_len] image
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(3,3), padding=(1,1))
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=(3,3), padding=(1,1))
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.conv3 = torch.nn.Conv2d(128, 256, kernel_size=(3,3), padding=(1,1))
        self.bn3 = torch.nn.BatchNorm2d(256)
        self.conv4 = torch.nn.Conv2d(256, out_dim, kernel_size=(3,3), padding=(1,1))
        self.bn4 = torch.nn.BatchNorm2d(out_dim)
        self.relu = torch.nn.ReLU()
        # only pool over length dimension, keep channel dim intact
        # self.pool2d = torch.nn.MaxPool2d(kernel_size=(1,2))
        self.pool2d = torch.nn.MaxPool2d(kernel_size=(2,2))  # IGNORE
        self.global_pool = torch.nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        # x: [batch, in_channels, in_len] -> [batch,1,in_channels,in_len]
        x = x.unsqueeze(1)
        # conv1 -> bn -> relu -> pool
        x = self.pool2d(self.relu(self.bn1(self.conv1(x)))) 
        # conv2 -> bn -> relu -> pool
        x = self.pool2d(self.relu(self.bn2(self.conv2(x))))
        # conv3 -> bn -> relu -> pool
        x = self.pool2d(self.relu(self.bn3(self.conv3(x))))
        # conv4 -> bn -> relu -> global pool
        x = self.global_pool(self.relu(self.bn4(self.conv4(x))))  # [batch, out_dim, 1, 1]
        x = x.view(x.size(0), -1)    # [batch, out_dim]
        # L2 normalize output
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-6)
        return x                     # [batch, out_dim]

def load_model(pretrained=False, ckpt_path=None, out_dim=512):
    model = ForceModelCNN(out_dim=out_dim)
    if pretrained:
        if ckpt_path is None:
            raise ValueError("ckpt_path must be provided when pretrained is True")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint file {ckpt_path} does not exist.")
        model.load_state_dict(torch.load(ckpt_path))
    return model