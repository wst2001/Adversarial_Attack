import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
def FGSM(img, gradients, episilon):
    perturbed_img = img + gradients.sign() * (episilon + random.uniform(-episilon/10, episilon/10))
    perturbed_img = torch.clamp(perturbed_img, 0, 1)
    return perturbed_img
def FGSM_attack(model, data_loader, episilon):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    criterion = nn.NLLLoss()
    model.to(device)
    criterion = criterion.to(device)
    attack_cnt = 0
    test_cnt = 0
    model.eval()
    for inputs, labels in data_loader:
        inputs.to(device)
        labels.to(device)
        inputs.requires_grad = True
        outputs = model(inputs)
        _, predicts = torch.max(outputs, 1)
        if (predicts != labels):
            attack_cnt += 1
            continue
        else:
            test_cnt += 1
        loss = criterion(outputs, labels)
        model.zero_grad()
        loss.backward()
        gradients = inputs.grad.data

        attack_inputs = FGSM(inputs, gradients, episilon)
        attack_outputs = model(attack_inputs)
        _, predicts = torch.max(attack_outputs, 1)
        if (predicts != labels):
            attack_cnt += 1
    test_acc = test_cnt / len(data_loader)
    attack_acc = attack_cnt / len(data_loader)
    print("Test Accuracy: %.4f" % test_acc)
    print("FGSM Attack Accuracy: %.4f" % attack_acc)




