import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def PGD_attack(model, data_loader, iters, episilon, eps_per_iter):
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
        adv = torch.zeros_like(inputs)

        for i in range(iters):
            loss = criterion(outputs, labels)
            model.zero_grad()
            loss.backward()
            gradients = inputs.grad.data
            adv += gradients.sign() * eps_per_iter
            adv = torch.clamp(adv, -episilon, episilon)
            attack_inputs = inputs + adv
            outputs = model(attack_inputs)
            _, predicts = torch.max(outputs, 1)
            if (predicts != labels):
                attack_cnt += 1
                break
    test_acc = test_cnt / len(data_loader)
    attack_acc = attack_cnt / len(data_loader)
    print("Test Accuracy: %.4f" % test_acc)
    print("FGSM Attack Accuracy: %.4f" % attack_acc)