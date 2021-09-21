import torch
import os
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import matplotlib.pyplot as plt

def save_model(model, path):
    torch.save(model.state_dict(), path)


def save_checkpoint(checkpoint, path):
    torch.save(checkpoint, path)


def load_model(model, path):
    print("Loading Model...")
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    print("Model Loaded Correctly")
    return model


def get_train_history(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    epoch = checkpoint['epoch']
    loss_history = checkpoint['loss_history']
    val_history = checkpoint['val_history']
    return epoch, loss_history, val_history


def load_checkpoint(model, optimizer, path):
    print("Loading Checkpoint...")
    checkpoint = torch.load(path, map_location=DEVICE)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_history = checkpoint['loss_history']
    val_history = checkpoint['val_history']
    print("Checkpoint Loaded Correctly!")
    return epoch, model, optimizer, loss_history, val_history