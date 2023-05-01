import torch
import torch.nn as nn
from torch.utils.data import (DataLoader, TensorDataset)
import os
import torch
import time
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import (Dataset, DataLoader, random_split)
from torch.utils.data import DataLoader
import pandas as pd
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics import Accuracy
import test_lib
from importlib import reload
from torch.nn import functional as F
reload(test_lib)
import warnings
warnings.filterwarnings('ignore')

def test_saved_model(model = None):
    device = torch.device('cpu')
    train_transforms = transforms.Compose([
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Grayscale(),
])

    #test_dataset = torch.load('./test_dataset.npz')
    test_dataset = datasets.ImageFolder('test', transform=train_transforms, is_valid_file=lambda x: os.path.splitext(x)[1] in ['.jpg', '.jpeg', '.png'])
    if model is None:

        model = torch.load('./mymodel.pt').to(device)
    loss = nn.CrossEntropyLoss()
    dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    acc = 0
    with torch.no_grad():
        for xs, targets in dataloader:
            xs, targets = xs.to(device), targets.to(device)
            ys = model(xs)
            acc += (ys.argmax(axis=1) == targets).sum().item()
    acc = acc / len(test_dataset) * 100
    print("Saved model has test accuracy = %.2f" % acc)

