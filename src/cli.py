from model import small_cnn, PreActResNet18
from src import FGSM
from src import PGD
from src import ZOO
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
def datasets_init():
    General_Transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    MNIST_dataset = MNIST(root='./datasets', train=False, transform=General_Transform, download=True)
    MNIST_dataset, _ = torch.utils.data.random_split(MNIST_dataset, [1000, 9000])
    
    #CIFAR10_dataset = CIFAR10(root='./datasets', train=False, transform=General_Transform, download=True)
    BATCH_SIZE = 1
    NUM_WORKERS = 1
    MNIST_loader = DataLoader(MNIST_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    #CIFAR10_loader = DataLoader(CIFAR10_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    return MNIST_loader#, CIFAR10_loader


def model_init():
    MNIST_model = small_cnn.create_network()
    MNIST_model.load_state_dict(torch.load('./checkpoint/MNIST_small_cnn.checkpoint')['state_dict'])
    #CIFAR10_model = PreActResNet18.PreActResNet18()
    #CIFAR10_model.load_state_dict(torch.load('./checkpoint/CIFAR10_PreActResNet18.checkpoint')['state_dict'])
    return MNIST_model#, CIFAR10_model
def run():
    MNIST_model = model_init()
    MNIST_loader= datasets_init()
    #FGSM.FGSM_attack(MNIST_model, MNIST_loader, episilon=0.3) # MNIST
    #FGSM.FGSM_attack(CIFAR10_model, CIFAR10_loader, episilon=0.031) #CIFAR10
    #PGD.PGD_attack(CIFAR10_model, CIFAR10_loader, iters=10, episilon=0.031, eps_per_iter=0.006)
    #ZOO.Simple_Gradient_Attack(MNIST_model, MNIST_loader, iters=1000, episilon=0.3, eps=1)
    ZOO.zoo_Adam(MNIST_model, MNIST_loader, iters=1000, episilon=0.3)
    
