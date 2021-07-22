import torch
import os


def save_model(model, path):
    torch.save(model.state_dict(), path)


def save_checkpoint(checkpoint, path):
    torch.save(checkpoint, path)


def load_model():
    pass


def load_checkpoint():
    pass