import torch
import os
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def save_model(model, path):
    torch.save(model.state_dict(), path)


def save_checkpoint(checkpoint, path):
    torch.save(checkpoint, path)


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_history = checkpoint['loss_history']
    val_history = checkpoint['val_history']
    return epoch, model, optimizer, loss_history, val_history
