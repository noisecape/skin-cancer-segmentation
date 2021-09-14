import torch
import os
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import matplotlib.pyplot as plt

def save_model(model, path):
    torch.save(model.state_dict(), path)


def save_checkpoint(checkpoint, path):
    torch.save(checkpoint, path)


def load_model(model, path):
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def get_train_history(path):
    checkpoint = torch.load(path, map_location=DEVICE)
    epoch = checkpoint['epoch']
    loss_history = checkpoint['loss_history']
    val_history = checkpoint['val_history']
    return epoch, loss_history, val_history


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    loss_history = checkpoint['loss_history']
    val_history = checkpoint['val_history']
    return epoch, model, optimizer, loss_history, val_history


# def visualize_final_results(path):
#     absolute_path = 'C:/Users/Noisecape/PycharmProjects/skin-cancer-segmentation/saved_models'
#     for p in path:
#         p = os.path.join(absolute_path, p)
#         checkpoint = torch.load(p, map_location=DEVICE)
#         loss_history = checkpoint['loss_history']
#         plt.plot(loss_history[:2000])
#     plt.title(f'Train Loss vs Validation Loss - {"Experiment 1 - Results"}')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.yscale('log')
#
#     plt.legend()
#     plt.show()
#
#
# absolute_path = 'C:/Users/Noisecape/PycharmProjects/skin-cancer-segmentation/saved_models'
# paths = [el if '.pth' in el else 0 for el in os.listdir(absolute_path)]
# while 0 in paths:
#     paths.remove(0)
# visualize_final_results(paths)