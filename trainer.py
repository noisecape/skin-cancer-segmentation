"""
This script contains all of the functions required to train
and run the experiments for each model.
"""

import torch.nn
import torch
from data.pretext_datasets import ContextRestorationDataPretext, ContrastiveLearningDataPretext
from data.pretext_datasets import CustomDataPretext, JiGenData
from torch.utils.data import DataLoader
from data.segmentation_dataset import SegmentationDataset
from model.context_restoration import UNET
from model.contrastive_learning import SimCLR
from model.jigsaw import JiGen
from model.utils.criterions import ContrastiveLoss
from model.custom_approach import CustomSegmentation
from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import random
from PIL import Image
import os
import torchvision
import matplotlib.pyplot as plt
from model.utils import utility
from model.utils.utility import load_model, load_checkpoint
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

saved_files_path = os.path.join(os.curdir, 'saved_models')


def get_jigen_data(P, batch_size, split, conf):
    """
    This function initializes the JiGen algorithm. If the checkpoints and the weights
    for the final model are provided with the ".pth" files, then those values are loaded
    into the model.
    :param P: the number of Jigsaw puzzles to create per each image.
    :param batch_size: the size of each batch
    :param split: the split percentage used to split the collection of images.
    :return data, phase: data is a dictionary containing all the relevant information for the model,
    phase indicates the current phase of the training process (either "train" or "test").
    """
    model = JiGen(conf=conf, P=P).to(DEVICE)
    train_data = JiGenData(P=P, mode='train', split=split)
    val_data = JiGenData(P=P, mode='val', split=split)
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader_train = DataLoader(train_data, **dataloader_params)
    dataloader_val = DataLoader(val_data, **dataloader_params)
    criterion_pretext = torch.nn.CrossEntropyLoss()
    criterion_segmentation = torch.nn.BCEWithLogitsLoss()
    optimizer = model.optimizer
    epoch = 0
    loss_history = []
    val_history = []
    phase = 'train'
    if os.path.exists(os.path.join(saved_files_path, 'jigen_model.pth')):
        model_path = os.path.join(saved_files_path, 'jigen_model.pth')
        model = load_model(model, path=model_path)
        phase = 'test'
    elif os.path.exists(os.path.join(saved_files_path, 'jigen_checkpoint.pth')):
        model_path = os.path.join(saved_files_path, 'jigen_checkpoint.pth')
        epoch, model, optimizer, loss_history, val_history = load_checkpoint(model, optimizer, model_path)
    data = {'model': model, 'optimizer': optimizer,
            'train_loader': dataloader_train, 'val_loader': dataloader_val,
            'criterion_seg': criterion_segmentation, 'criterion_ptx': criterion_pretext,
            'epoch': epoch, 'loss_history': loss_history, 'val_history': val_history}
    return data, phase


def get_supervised_data(batch_size, split, conf):
    """
    This function initializes the U-Net. If the checkpoints and the weights
    for the final model are provided with the ".pth" files, then those values are loaded
    into the model.
    :param batch_size: the size of each batch
    :param split: the split percentage used to split the collection of images
    :return data, phase: data is a dictionary containing all the relevant information for the model,
    phase indicates the current phase of the training process (either "train" or "test").
    """
    model = UNET(in_channels=3, name='supervised').to(DEVICE)
    train_data = SegmentationDataset(mode='train', split_perc=split)
    val_data = SegmentationDataset(mode='val', split_perc=split)
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader_train = DataLoader(train_data, **dataloader_params)
    dataloader_val = DataLoader(val_data, **dataloader_params)
    criterion = torch.nn.BCEWithLogitsLoss()

    if conf == 1:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    elif conf == 2:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.01, momentum=0.995, nesterov=True)
    elif conf == 3:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    epoch = 0
    loss_history = []
    val_history = []
    phase = 'train'
    if os.path.exists(os.path.join(saved_files_path, 'segmentation_model_supervised.pth')):
        model_path = os.path.join(saved_files_path, 'segmentation_model_supervised.pth')
        model = load_model(model, path=model_path)
        phase = 'test'
    elif os.path.exists(os.path.join(saved_files_path, 'segmentation_checkpoint_supervised.pth')):
        model_path = os.path.join(os.path.join(saved_files_path, 'segmentation_checkpoint_supervised.pth'))
        epoch, model, optimizer, loss_history, val_history = load_checkpoint(model, optimizer, model_path)
    data = {'model': model, 'optimizer': optimizer,
            'train_loader': dataloader_train, 'val_loader': dataloader_val,
            'criterion': criterion, 'epoch': epoch,
            'loss_history': loss_history, 'val_history': val_history}
    return data, phase


def get_contrastive_learning_pretext(batch_size, full_data, conf):
    """
    This function initializes SimCLR. If the checkpoints and the weights
    for the final model are provided with the ".pth" files, then those values are loaded
    into the model.
    :param batch_size: the size of each batch
    :param full_data: if set to True, then considers the entire HAM10000 collection from the "Resized_All" folder.
    Otherwise, it considers the small subset of unlabelled images from the "Resized" folder.
    :return data, phase: data is a dictionary containing all the relevant information for the model,
    phase indicates the current phase of the training process (either "train" or "test").
    """
    model = SimCLR().to(DEVICE)
    train_data = ContrastiveLearningDataPretext(mode='train', full_data=full_data)
    val_data = ContrastiveLearningDataPretext(mode='val', full_data=full_data)
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader_train = DataLoader(train_data, **dataloader_params)
    dataloader_val = DataLoader(val_data, **dataloader_params)
    criterion = ContrastiveLoss().to(DEVICE)

    if conf == 1:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01, amsgrad=True)
    elif conf == 2:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.995, weight_decay=0.05, nesterov=True)
    elif conf == 3:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.995, weight_decay=0.05, nesterov=True)
    elif conf == 4:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.995, weight_decay=0.05, nesterov=True)

    epoch = 0
    loss_history = []
    val_history = []
    phase = 'pretext'
    # check if the model has already been trained
    if os.path.exists(os.path.join(saved_files_path, 'contrastive_model_pretext.pth')):
        model_path = os.path.join(saved_files_path, 'contrastive_model_pretext.pth')
        model = load_model(model, path=model_path)
        phase = 'segmentation'
    elif os.path.exists(os.path.join(saved_files_path, 'contrastive_checkpoint_pretext.pth')):
        model_path = os.path.join(saved_files_path, 'contrastive_checkpoint_pretext.pth')
        epoch, model, optimizer, loss_history, val_history = load_checkpoint(model, optimizer, model_path)
    data = {'model': model, 'optimizer': optimizer,
            'train_loader': dataloader_train, 'val_loader': dataloader_val,
            'criterion': criterion, 'epoch': epoch,
            'loss_history': loss_history, 'val_history': val_history}
    return data, phase


def get_context_restoration_pretext(T, batch_size, full_data, conf):
    """
    Intializes the context restoration model for the pretext task. If the checkpoints and the weights
    for the final model are provided with the ".pth" files, then those values are loaded
    into the model.
    :param T: represents the number of pairs that need to be swapped while corrupting the image.
    :param batch_size: the size of each batch
    :param full_data: if set to True, then considers the entire HAM10000 collection from the "Resized_All" folder.
    Otherwise, it considers the small subset of unlabelled images from the "Resized" folder.
    :return data, phase: data is a dictionary containing all the relevant information for the model,
    phase indicates the current phase of the training process (either "train" or "test").
    """
    model = UNET(in_channels=3).to(DEVICE)
    train_data = ContextRestorationDataPretext(T=T, mode='train', full_data=full_data)
    val_data = ContextRestorationDataPretext(T=T, mode='val', full_data=full_data)
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader_train = DataLoader(train_data, **dataloader_params)
    dataloader_val = DataLoader(val_data, **dataloader_params)
    criterion = torch.nn.MSELoss()

    if conf == 1:
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01, lr=0.0001)
    elif conf == 2:
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.01, lr=0.0001, momentum=0.99, nesterov=True)
    elif conf == 3:
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.01, lr=0.001, momentum=0.99, nesterov=True)
    elif conf == 4:
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.05, lr=0.001, momentum=0.995, nesterov=True)

    epoch = 0
    loss_history = []
    val_history = []
    phase = 'pretext'
    if os.path.exists(os.path.join(saved_files_path, 'context_model_pretext.pth')):
        model_path = os.path.join(saved_files_path, 'context_model_pretext.pth')
        model = load_model(model, path=model_path)
        phase = 'fine-tune'
    elif os.path.exists(os.path.join(saved_files_path, 'context_checkpoint_pretext.pth')):
        model_path = os.path.join(saved_files_path, 'context_checkpoint_pretext.pth')
        epoch, model, optimizer, loss_history, val_history = load_checkpoint(model, optimizer, model_path)

    data = {'model': model, 'optimizer': optimizer,
            'train_loader': dataloader_train, 'val_loader': dataloader_val,
            'criterion': criterion, 'epoch': epoch,
            'loss_history': loss_history, 'val_history': val_history}
    return data, phase


def get_custom_approach_pretext(batch_size, full_data, conf):
    """
    This function initializes Personal Model for the pretext task. If the checkpoints and the weights
    for the final model are provided with the ".pth" files, then those values are loaded
    into the model.
    :param batch_size: the size of each batch.
    :param full_data: if set to True, then considers the entire HAM10000 collection from the "Resized_All" folder.
    Otherwise, it considers the small subset of unlabelled images from the "Resized" folder.
    :return data, phase: data is a dictionary containing all the relevant information for the model,
    phase indicates the current phase of the training process (either "train" or "test").
    """
    model = CustomSegmentation(n_augmentations=4).to(DEVICE)
    train_data = CustomDataPretext(mode='train', full_data=full_data)
    val_data = CustomDataPretext(mode='val', full_data=full_data)
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader_train = DataLoader(train_data, **dataloader_params)
    dataloader_val = DataLoader(val_data, **dataloader_params)
    criterion = torch.nn.CrossEntropyLoss()

    if conf == 1:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
    elif conf == 2:
        optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.01, lr=0.001, momentum=0.995, nesterov=True)
    elif conf == 3:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    elif conf == 4:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    epoch = 0
    loss_history = []
    val_history = []
    phase = 'pretext'
    if os.path.exists(os.path.join(saved_files_path, 'personal_model_pretext.pth')):
        model_path = os.path.join(saved_files_path, 'personal_model_pretext.pth')
        model = load_model(model, path=model_path)
        phase = 'segmentation'
    elif os.path.exists(os.path.join(saved_files_path, 'personal_checkpoint_pretext.pth')):
        model_path = os.path.join(saved_files_path, 'personal_checkpoint_pretext.pth')
        epoch, model, optimizer, loss_history, val_history = load_checkpoint(model, optimizer, model_path)
    data = {'model': model, 'optimizer': optimizer,
            'train_loader': dataloader_train, 'val_loader': dataloader_val,
            'criterion': criterion, 'epoch': epoch,
            'loss_history': loss_history, 'val_history': val_history}
    return data, phase


def get_segmentation(model, optimizer, batch_size, technique, split):
    """
    This function builds the dataset for the segmentation task. It also initializes all the components
    to be able to run an experiment.
    :param model: if there is a model to be loaded, the values of the weighs of this model will be loaded from the
    ".pth" file.
    :param optimizer: the optimizer used to train the model. If a checkpoint is given, then it values will be loaded
    form the ".pth" file.
    :param batch_size: the size of each batch
    :param technique: the name of the technique that is currently executing the experiment. Possible values are:
    'context_restoration', 'contrastive_learning', 'jigen', 'custom_approach'.
    :param split: the split percentage used to split the dataset
    :return data, phase: data is a dictionary containing all the relevant information for the model,
    phase indicates the current phase of the training process (either "fine-tune" or "completed").
    """
    train_data = SegmentationDataset(mode='train', split_perc=split)
    val_data = SegmentationDataset(mode='val', split_perc=split)
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    train_loader = DataLoader(train_data, **dataloader_params)
    val_loader = DataLoader(val_data, **dataloader_params)
    criterion = torch.nn.BCEWithLogitsLoss()
    epoch = 0
    loss_history = []
    val_history = []
    phase = 'fine-tune'
    if technique == 'jigsaw':

        if os.path.exists(os.path.join(saved_files_path, 'segmentation_model_jigsaw.pth')):
            model_path = os.path.join(saved_files_path, 'segmentation_model_jigsaw.pth')
            model = load_model(model, path=model_path)
            phase = 'completed'
        elif os.path.exists(os.path.join(saved_files_path, 'segmentation_checkpoint_jigsaw.pth')):
            model_path = os.path.join(saved_files_path, 'segmentation_checkpoint_jigsaw.pth')
            epoch, model, optimizer, loss_history, val_history = load_checkpoint(model, optimizer, model_path)

    elif technique == 'contrastive_learning':

        if os.path.exists(os.path.join(saved_files_path, 'segmentation_model_contrastive.pth')):
            model_path = os.path.join(saved_files_path, 'segmentation_model_contrastive.pth')
            model = load_model(model, path=model_path)
            phase = 'completed'
        elif os.path.exists(os.path.join(saved_files_path, 'segmentation_checkpoint_contrastive.pth')):
            model_path = os.path.join(saved_files_path, 'segmentation_checkpoint_contrastive.pth')
            epoch, model, optimizer, loss_history, val_history = load_checkpoint(model, optimizer, model_path)

    elif technique == 'context_restoration':

        if os.path.exists(os.path.join(saved_files_path, 'segmentation_model_restoration.pth')):
            model_path = os.path.join(saved_files_path, 'segmentation_model_restoration.pth')
            model = load_model(model, path=model_path)
            phase = 'completed'
        elif os.path.exists(os.path.join(saved_files_path, 'segmentation_checkpoint_restoration.pth')):
            model_path = os.path.join(saved_files_path, 'segmentation_checkpoint_restoration.pth')
            epoch, model, optimizer, loss_history, val_history = load_checkpoint(model, optimizer, model_path)

    elif technique == 'custom_approach':
        if os.path.exists(os.path.join(saved_files_path, 'segmentation_model_custom_approach.pth')):
            model_path = os.path.join(saved_files_path, 'segmentation_model_custom_approach.pth')
            model = load_model(model, path=model_path)
            phase = 'completed'
        elif os.path.exists(os.path.join(saved_files_path, 'segmentation_checkpoint_custom_approach.pth')):
            model_path = os.path.join(saved_files_path, 'segmentation_checkpoint_custom_approach.pth')
            epoch, model, optimizer, loss_history, val_history = load_checkpoint(model, optimizer, model_path)

    data = {'model': model, 'optimizer': optimizer,
            'train_loader': train_loader, 'val_loader': val_loader,
            'criterion': criterion, 'epoch': epoch,
            'loss_history': loss_history, 'val_history': val_history}
    return data, phase


def get_eval_dataset(split):
    """
    Returns the split used for the evaluation phase.
    :param split: the split percentage.
    :return dataloader: the dataloader used to load the data for the testing phase
    """
    dataset = SegmentationDataset(mode='test', split_perc=split)
    dataloader_params = {'shuffle': True, 'batch_size': 1}
    dataloader = DataLoader(dataset, **dataloader_params)
    return dataloader


class Trainer(ABC):
    """
    Represents the abstract class that contains all of the functionalities that
    each trainer has to implements. A trainer is the object that implements all the functionalities
    required to run a specific experiment.
    """

    def __init__(self, n_epochs_pretext, n_epochs_segmentation, technique):
        self.n_epochs_pretext = n_epochs_pretext
        self.n_epochs_segmentation = n_epochs_segmentation
        self.technique = technique

    @abstractmethod
    def train_pretext(self, train_loader, val_loader, model, optimizer, criterion, epoch, loss_history, val_history):
        pass

    @abstractmethod
    def train_batch(self, train_loader, permutation_set, model, criterion, optimizer):
        pass

    @abstractmethod
    def validate(self, val_loader, permutation_set, model, criterion):
        pass

    def train_segmentation(self, train_loader, val_loader, model, optimizer, criterion, epoch, loss_history, val_history):
        """
        Implements the train loop for the segmentation task. This loop is in common with all of the self-supervised
        algorithms, excluding JiGen.
        :param train_loader: the dataloader to load the training samples
        :param val_loader: the dataloader to load the validation samples
        :param model: the model that is currently training (SimCLR, Personal Model or Context Restoration)
        :param optimizer: the optimizer used to calculate the gradients and update the weights of the model
        :param criterion: the criterion used to calculate the loss
        :param epoch: the number of epochs already processed.
        :param loss_history: history of the training loss
        :param val_history: history of the validation loss.
        """
        loop = tqdm(range(epoch, self.n_epochs_segmentation), total=self.n_epochs_segmentation-epoch, leave=False)
        # load the model from pretext task
        for e in loop:
            model.train()
            running_loss_train = []
            running_loss_val = []
            # train
            for idx, (imgs, gt_imgs) in enumerate(train_loader):
                prediction = model(imgs.to(DEVICE), pretext=False)
                loss = criterion(prediction, gt_imgs.to(DEVICE))
                running_loss_train.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print statistics
            batch_loss = torch.sum(torch.tensor(running_loss_train))/len(train_loader)
            loss_history.append(batch_loss.item())
            model.eval()
            with torch.no_grad():
                # validation
                for idx, (imgs, gt_imgs) in enumerate(val_loader):
                    prediction = model(imgs.to(DEVICE), pretext=False)
                    loss = criterion(prediction, gt_imgs.to(DEVICE))
                    running_loss_val.append(loss)
                batch_loss = torch.sum(torch.tensor(running_loss_val))/len(val_loader)
                val_history.append(batch_loss.item())
                loop.set_description('Epoch segmentation processed')
                loop.set_postfix(train_loss=loss_history[-1], val_loss=val_history[-1])
            # save checkpoint
            checkpoint = {
                'epoch': e,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_history': loss_history,
                'val_history': val_history
            }
            utility.save_checkpoint(checkpoint,
                                    path=os.path.join(os.curdir,
                                                      'saved_models/segmentation_checkpoint'+f'_{self.technique}'+'.pth'))
        # save the model
        utility.save_model(model, path=os.path.join(os.curdir, 'saved_models/segmentation_model'+f'_{self.technique}'+'.pth'))

    def evaluate(self, dataloader, model, p_threshold, T):
        """
        Implements the testing loop. This loop is in common with all of the self-supervised
        algorithms, excluding JiGen.
        :param dataloader: the dataloader used to load the data from the testing split.
        :param model: the model to validated.
        :param p_threshold: the threshold used to classify each pixel as background or lesion.
        :param T: the threshold used to implement the Thresholded_IoU.
        :return accuracy, accuracy_no_t, dice_score, sensitivity, specificity:
        """
        output_path = os.path.join(os.curdir, f'saved_models/predictions_{model.name}')
        prediction_samples = []
        model.eval()
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        ious = []
        ious_no_threshold = []
        dice_score = []
        sensitivity = []
        specificity = []
        with torch.no_grad():
            for idx, batch in loop:
                img = batch[0].to(DEVICE)
                gt = batch[1].to(DEVICE)
                output = torch.sigmoid(model(img, pretext=False)).squeeze(0)
                prediction = (output > p_threshold).float()
                gt = gt.squeeze(0)
                if random.random() > 0.5 and len(prediction_samples) < 10:
                    prediction_samples.append((gt, prediction))
                    torchvision.utils.save_image(gt, os.path.join(output_path, f'GT_{idx}.png'))
                    torchvision.utils.save_image(prediction, os.path.join(output_path, f'PRED_{idx}.png'))
                prediction = prediction.view(-1)
                gt = gt.view(-1)
                true_positives = torch.sum(torch.tensor([1.0 if p == 1.0 and g == 1.0 else 0.0
                                                         for p, g in zip(prediction, gt)]))
                true_negatives = torch.sum(torch.tensor([1.0 if p == 0.0 and g == 0.0 else 0.0
                                                         for p, g in zip(prediction, gt)]))
                false_positives = torch.sum(torch.tensor([1.0 if p == 1.0 and g == 0.0 else 0.0
                                                         for p, g in zip(prediction, gt)]))
                false_negatives = torch.sum(torch.tensor([1.0 if p == 0.0 and g == 1.0 else 0.0
                                                         for p, g in zip(prediction, gt)]))
                iou = true_positives / (true_positives + false_negatives + false_positives)
                ious.append((iou.float() if iou > T else 0.0))
                ious_no_threshold.append(iou.float())
                dice_score.append(((2 * true_positives) / ((2 * true_positives) + false_negatives + false_positives)).item())
                sensitivity.append((true_positives / (true_positives + false_negatives)).item())
                specificity.append((true_negatives / (true_negatives + false_positives)).item())
                loop.set_postfix(t_IoU=np.sum(ious) / len(ious),
                                 IoU=np.sum(ious_no_threshold) / len(ious_no_threshold),
                                 dice_score=np.sum(dice_score) / len(dice_score),
                                 sensitivity=np.sum(sensitivity) / len(sensitivity),
                                 specificity=np.sum(specificity) / len(specificity))

        accuracy = np.sum(ious) / len(ious)
        accuracy_no_t = np.sum(ious_no_threshold) / len(ious_no_threshold)
        dice_score = np.sum(dice_score) / len(dice_score)
        sensitivity = np.sum(sensitivity) / len(sensitivity)
        specificity = np.sum(specificity) / len(specificity)
        return accuracy, accuracy_no_t, dice_score, sensitivity, specificity


    def visualize_prediction(self, model, dataloader, p_threshold):
        """
        Plot the prediction given in output by the model.
        :param model: the model used to produced the output.
        :param dataloader: the dataloader used to load the testing data.
        :param p_threshold: the threshold used to classify each pixel as background or lesion.
        :return:
        """
        model.eval()
        with torch.no_grad():
            sample = random.randint(0, len(dataloader.dataset))
            x, mask = dataloader.dataset[sample]
            prediction = torch.sigmoid(model(x.unsqueeze(0).to(DEVICE))).squeeze(0)
            prediction[prediction > p_threshold] = 255.0
            prediction[prediction < p_threshold] = 0.0

            plt.figure(figsize=(16, 16))
            prediction = prediction.permute(1, 2, 0)
            plt.imshow(prediction.cpu())
            plt.axis('off')
            plt.show()
            plt.close()


class SupervisedTrainer(Trainer):

    """
    This class implements the trainer for the supervised model, that is the U-Net
    """

    def __init__(self, n_epochs_segmentation, technique, n_epochs_pretext=0):
        super(SupervisedTrainer, self).__init__(n_epochs_pretext, n_epochs_segmentation, technique)
        self.n_epochs_segmentation = n_epochs_segmentation
        self.technique = technique

    def train_pretext(self, train_loader, val_loader, model, optimizer, criterion, epoch, loss_history, val_history):
        pass

    def train_batch(self, train_loader, permutation_set, model, criterion, optimizer):
        pass

    def validate(self, val_loader, permutation_set, model, criterion):
        pass


class JigenTrainer:

    """
    This class implements the trainer for JiGen
    """

    def __init__(self, n_epochs, P, N, batch_size, technique='jigen', alpha=0.9, beta=0.6):
        self.P = P
        self.N = N
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.technique = technique
        self.alpha = alpha
        self.beta = beta

    def train(self, model, train_loader, val_loader, optimizer, criterion_ptx, criterion_seg, epoch, loss_history, val_history):
        """
        Implements the training loop for JiGen. The training loop is composed by two phases: firstly, the pretext
        submodel is used to optimize the pretext task; secondly, the segmentation submodel is used to optimize the
        segmentation task. The total loss is the sum of the two individual losses coming from the previous phases.
        :param model: the model that is currently being trained.
        :param train_loader: the dataloader to load the data from the training split.
        :param val_loader: the dataloader to load the data from the validation split.
        :param optimizer: the optimizer used in the training process.
        :param criterion_ptx: the criterion used to calculate the loss for the pretext task
        :param criterion_seg: the criterion used to calculate the loss for the segmentation task.
        :param epoch: the number of epochs already processed.
        :param loss_history: the history of the training loss.
        :param val_history: the history of the validation loss
        :return:
        """
        loop = tqdm(range(epoch, self.n_epochs), total=self.n_epochs - epoch, leave=False)
        for e in loop:
            batch_loss = self.train_batch(model, train_loader, criterion_seg, criterion_ptx, optimizer)
            loss_history.append(batch_loss.item())
            val_loss = self.validate(model, val_loader, criterion_seg, criterion_ptx)
            val_history.append(val_loss.item())
            loop.set_description(f'Epoch')
            loop.set_postfix(train_loss=loss_history[-1], val_loss=val_history[-1])
            # checkpoint
            checkpoint = {
                'epoch': e,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_history': loss_history,
                'val_history': val_history
            }
            utility.save_checkpoint(checkpoint, path=os.path.join(os.curdir,
                                                                  'saved_models/jigen_checkpoint.pth'))
        # save the model
        utility.save_model(model, path=os.path.join(os.curdir, 'saved_models/jigen_model.pth'))

    def train_batch(self, model, data_loader, criterion_seg, criterion_ptx, optimizer):
        """
        Execute one training epoch over a specific batch.
        :param model: the model currently being trained.
        :param data_loader: the dataloader used to load the data from the training split.
        :param criterion_seg: the criterion used to calculate the loss for the segmentation task
        :param criterion_ptx: the criterion used to calculate the loss for the pretext task.
        :param optimizer: the optimizer used during the training.
        :return:
        """
        model.train()
        batch_loss = []
        for ptx_img, labels, imgs_seg, segs_gt in data_loader:
            permutation_loss = 0
            ptx_img = ptx_img.permute((1, 0, 2, 3, 4)).to(DEVICE)
            labels = labels.permute((1, 0)).to(DEVICE)
            imgs_seg = imgs_seg.to(DEVICE)
            segs_gt = segs_gt.to(DEVICE)
            out_seg = model(imgs_seg, pretext=False)
            loss_seg = criterion_seg(out_seg, segs_gt)
            # process each jigsaw puzzle
            for n, (imgs, label) in enumerate(zip(ptx_img, labels)):
                out_ptx = model(imgs, pretext=True)
                pretext_loss = criterion_ptx(out_ptx, label)
                permutation_loss += (self.alpha*pretext_loss)
            permutation_loss = torch.sum(permutation_loss) / self.P
            # combine the two losses to obtain the final loss that is then backpropagated.
            total_loss = permutation_loss + loss_seg
            batch_loss.append(total_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        batch_loss = torch.sum(torch.tensor(batch_loss) / len(data_loader))
        return batch_loss

    def validate(self, model, val_loader, criterion_seg, criterion_ptx):
        """
        Implements the validation loop.
        :param model: the model currently being validated.
        :param val_loader: the dataloader used to load the data from the validation split.
        :param criterion_seg: the criterion used to calculate the loss for the segmentation task.
        :param criterion_ptx: the criterion used to calculate the loss for the pretext task.
        :return:
        """
        running_loss_eval = []
        model.eval()
        with torch.no_grad():
            for ptx_img, labels, imgs_seg, segs_gt in val_loader:
                permutation_loss = 0
                ptx_img = ptx_img.permute((1, 0, 2, 3, 4)).to(DEVICE)
                labels = labels.permute((1, 0)).to(DEVICE)
                imgs_seg = imgs_seg.to(DEVICE)
                segs_gt = segs_gt.to(DEVICE)
                out_seg = model(imgs_seg, pretext=False)
                loss_seg = criterion_seg(out_seg, segs_gt)
                for n, (imgs, label) in enumerate(zip(ptx_img, labels)):
                    out_ptx = model(imgs, pretext=True)
                    pretext_loss = criterion_ptx(out_ptx, label)
                    permutation_loss += pretext_loss
                permutation_loss = torch.sum(permutation_loss) / self.P
                total_loss = permutation_loss + loss_seg
                running_loss_eval.append(total_loss)
            running_loss_eval = torch.sum(torch.tensor(running_loss_eval) / len(val_loader))
        return running_loss_eval

    def evaluate(self, dataloader, model, p_threshold, T):
        """
        Implements the testing loop.
        :param dataloader: the dataloader used to load the data from the testing split.
        :param model: the model currently being tested
        :param p_threshold: the threshold used to classify each pixel as background or lesion.
        :param T: the threshold used to define the Thresholded_IoU
        :return accuracy, accuracy_no_t, dice_score, sensitivity, specificity:
        """
        output_path = os.path.join(os.curdir, f'saved_models/predictions_{model.name}')
        prediction_samples = []
        model.eval()
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        ious = []
        ious_no_threshold = []
        dice_score = []
        sensitivity = []
        specificity = []
        with torch.no_grad():
            for idx, batch in loop:
                img = batch[2].to(DEVICE)
                # gt = batch[1].squeeze(1).view(-1).to(DEVICE)
                gt = batch[3].to(DEVICE)
                output = torch.sigmoid(model(img, pretext=False)).squeeze(0)
                prediction = (output > p_threshold).float()
                gt = gt.squeeze(0)
                if random.random() > 0.5 and len(prediction_samples) < 10:
                    prediction_samples.append((gt, prediction))
                    torchvision.utils.save_image(gt, os.path.join(output_path, f'GT_{idx}.png'))
                    torchvision.utils.save_image(prediction, os.path.join(output_path, f'PRED_{idx}.png'))
                prediction = prediction.view(-1)
                gt = gt.view(-1)
                true_positives = torch.sum(torch.tensor([1.0 if p == 1.0 and g == 1.0 else 0.0
                                                         for p, g in zip(prediction, gt)]))
                true_negatives = torch.sum(torch.tensor([1.0 if p == 0.0 and g == 0.0 else 0.0
                                                         for p, g in zip(prediction, gt)]))
                false_positives = torch.sum(torch.tensor([1.0 if p == 1.0 and g == 0.0 else 0.0
                                                         for p, g in zip(prediction, gt)]))
                false_negatives = torch.sum(torch.tensor([1.0 if p == 0.0 and g == 1.0 else 0.0
                                                         for p, g in zip(prediction, gt)]))
                iou = true_positives / (true_positives + false_negatives + false_positives)
                ious.append((iou.float() if iou > T else 0.0))
                ious_no_threshold.append(iou.float())
                sensitivity.append((true_positives / (true_positives + false_negatives)).item())
                specificity.append((true_negatives / (true_negatives + false_positives)).item())
                dice_score.append(((2 * true_positives) / ((2*true_positives) + false_negatives + false_positives)).item())
                loop.set_postfix(t_IoU=np.sum(ious) / len(ious),
                                 IoU=np.sum(ious_no_threshold) / len(ious_no_threshold),
                                 dice_score=np.sum(dice_score) / len(dice_score),
                                 sensitivity=np.sum(sensitivity) / len(sensitivity),
                                 specificity=np.sum(specificity) / len(specificity))

        accuracy = np.sum(ious) / len(ious)
        accuracy_no_t = np.sum(ious_no_threshold) / len(ious_no_threshold)
        dice_score = np.sum(dice_score) / len(dice_score)
        sensitivity = np.sum(sensitivity) / len(sensitivity)
        specificity = np.sum(specificity) / len(specificity)
        return accuracy, accuracy_no_t, dice_score, sensitivity, specificity


class ContrastiveLearningTrainer(Trainer):

    """
    This class implements the trainer for SimCLR
    """

    def __init__(self, n_epochs_pretext, n_epochs_segmentation):
        super(ContrastiveLearningTrainer, self).__init__(n_epochs_pretext, n_epochs_segmentation, technique='contrastive')
        self.n_epochs_pretext = n_epochs_pretext
        self.n_epochs_segmentation = n_epochs_segmentation

    def train_batch(self, train_loader, model, criterion, optimizer):
        """
        Implements the training loop for one batch.
        :param train_loader: the dataloader used to train the data from the training split
        :param model: the model currently being trained.
        :param criterion: the criterion used to calculate the loss for the pretext task.
        :param optimizer: the optimizer used to train the model.
        :return batch_loss: the total loss averaged by the number of elements in one batch.
        """
        running_loss = []
        for idx, x in enumerate(train_loader):
            processed_1, processed_2 = model(x.to(DEVICE), pretext=True)
            loss = criterion(processed_1.to(DEVICE), processed_2.to(DEVICE))
            running_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        batch_loss = torch.sum(torch.tensor(running_loss)) / len(train_loader)
        return batch_loss

    def validate(self, val_loader, model, criterion):
        """
        Implements the validation loop.
        :param val_loader: the dataloader used to load the data from the validation split
        :param model: the currently being validated.
        :param criterion: the criterion used to calulate the validation loss.
        :return: the validation loss averaged over the number of samples in the batch.
        """
        running_loss_eval = []
        with torch.no_grad():
            for idx, x in enumerate(val_loader):
                processed_1, processed_2 = model(x.to(DEVICE), pretext=True)
                loss = criterion(processed_1.to(DEVICE), processed_2.to(DEVICE))
                running_loss_eval.append(loss)
        return torch.sum(torch.tensor(running_loss_eval)) / len(val_loader)

    def train_pretext(self, train_loader, val_loader, model, optimizer, criterion, epoch, loss_history, val_history):
        """
        Implements the training loop for the pretext task.
        :param train_loader: the dataloader used to load the data from the training split
        :param val_loader: the dataloader used to load the data from the validation split
        :param model: the model currently being trained
        :param optimizer: the optimizer used to train the model
        :param criterion: the criterion used to calculate the loss for the pretext task
        :param epoch: the number of epochs already executed
        :param loss_history: the history of the training loss
        :param val_history: the history of the validation loss
        :return:
        """
        loop = tqdm(range(epoch, self.n_epochs_pretext), total=self.n_epochs_pretext - epoch, leave=False)
        for e in loop:
            model.train()
            # train loop
            batch_loss = self.train_batch(train_loader, model, criterion, optimizer)
            loss_history.append(batch_loss.item())
            # val loop
            model.eval()
            val_loss = self.validate(val_loader, model, criterion)
            val_history.append(val_loss.item())
            # print statistics
            loop.set_description(f'Epoch pretext processed')
            loop.set_postfix(train_loss=loss_history[-1], val_loss=val_history[-1])
            # save checkpoint
            checkpoint = {
                'epoch': e,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_history': loss_history,
                'val_history': val_history
            }
            utility.save_checkpoint(checkpoint, path=os.path.join(os.curdir, 'saved_models/contrastive_checkpoint_pretext.pth'))
        # save model
        utility.save_model(model, path=os.path.join(os.curdir, 'saved_models/contrastive_model_pretext.pth'))


class ContextRestorationTrainer(Trainer):
    """
    The class that implements the trainer for the Context Restoration model.
    """

    def __init__(self, n_epochs_pretext, n_epochs_segmentation):
        super(ContextRestorationTrainer, self).__init__(n_epochs_pretext, n_epochs_segmentation, technique='restoration')
        self.n_epochs_pretext = n_epochs_pretext
        self.n_epochs_segmentation = n_epochs_segmentation

    def train_batch(self, train_loader, model, criterion, optimizer):
        """
        Process one batch for the training loop.
        :param train_loader: the dataloader used to load the data from the training split.
        :param model: the model currently being tested.
        :param criterion: the criterion used to calculate the loss for the pretext task
        :param optimizer: the optimizer used to train the model.
        :return batch_loss: the averaged loss over the entire batch
        """
        running_loss = []
        for idx, batch in enumerate(train_loader):
            original = batch[0].to(DEVICE)
            corrupted = batch[1].to(DEVICE)
            restored = model(corrupted.to(DEVICE), pretext=True)
            loss = criterion(restored.to(DEVICE), original.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
        batch_loss = torch.sum(torch.tensor(running_loss)) / len(train_loader)
        return batch_loss

    def validate(self, val_loader, model, criterion):
        """
        Implements the validation loop
        :param val_loader: the dataloader used to load the data from the validation split
        :param model: the model currently being validated
        :param criterion: the criterion used to calculate the validation loss
        :return batch_loss: the validation loss averaged over the number of samples in the batch
        """
        running_loss_eval = []
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                original = batch[0].to(DEVICE)
                corrupted = batch[1].to(DEVICE)
                restored = model(corrupted.to(DEVICE), pretext=True)
                loss = criterion(restored.to(DEVICE), original.to(DEVICE))
                running_loss_eval.append(loss.item())
            batch_loss = torch.sum(torch.tensor(running_loss_eval)) / len(val_loader)
        return batch_loss

    def train_pretext(self, train_loader, val_loader, model, optimizer, criterion, epoch, loss_history, val_history):
        """
        Implements the training loop for the pretext task.
        :param train_loader: the dataloader used to load the data from the training split
        :param val_loader: the dataloader used to load the data from the validation split
        :param model: the model currently being trained
        :param optimizer: the optimizer used to train the model
        :param criterion: the criterion used to calculate the pretext loss
        :param epoch: the number of epochs already been processed
        :param loss_history: the history of the training loss
        :param val_history: the history of the validation loss
        :return:
        """
        loop = tqdm(range(epoch, self.n_epochs_pretext), total=self.n_epochs_pretext - epoch, leave=False)
        for e in loop:
            model.train()
            # train
            batch_loss = self.train_batch(train_loader, model, criterion, optimizer)
            loss_history.append(batch_loss.item())
            # validate
            model.eval()
            val_loss = self.validate(val_loader, model, criterion)
            val_history.append(val_loss.item())
            # print statistics
            loop.set_description(f'Epoch pretext processed')
            loop.set_postfix(train_loss=loss_history[-1], val_loss=val_history[-1])
            # save checkpoint
            checkpoint = {
                'epoch': e,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_history': loss_history,
                'val_history': val_history
            }
            utility.save_checkpoint(checkpoint, path=os.path.join(os.curdir, 'saved_models/context_checkpoint_pretext.pth'))
        # save model
        utility.save_model(model, path=os.path.join(os.curdir, 'saved_models/context_model_pretext.pth'))


class CustomApproachTrainer(Trainer):
    """
    This class implements the trainer for Personal Model
    """

    def __init__(self, n_epochs_pretext, n_epochs_segmentation):
        super(CustomApproachTrainer, self).__init__(n_epochs_pretext, n_epochs_segmentation, technique='custom_approach')
        self.n_epochs_pretext = n_epochs_pretext
        self.n_epochs_segmentation = n_epochs_segmentation

    def train_batch(self, train_loader, model, criterion, optimizer):
        """
        The loop that process a single batch during the training phase
        :param train_loader: the dataloader used to load the data from the training split
        :param model: the model currently being trained
        :param criterion: the criterion used to calculate the pretext loss
        :param optimizer: the optimizer used to train the model
        :return batch_loss: the loss averaged over the number of training samples in one batch
        """
        running_loss_train = []
        for idx, batch in enumerate(train_loader):
            images = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)
            output = model(images, pretext=True)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss_train.append(loss.item())
        # update train statistics
        batch_loss = torch.sum(torch.tensor(running_loss_train)) / len(train_loader)
        return batch_loss

    def validate(self, val_loader, model, criterion):
        """
        The validation loop
        :param val_loader: the dataloader used to load the data from the validation split
        :param model: the model currently being trained.
        :param criterion: the criterion used to calculate the validation loss
        :return batch_loss: the validation loss averaged over the number of samples in the batch
        """
        running_loss_eval = []
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                images = batch[0].to(DEVICE)
                labels = batch[1].to(DEVICE)
                output = model(images, pretext=True)
                loss = criterion(output, labels)
                running_loss_eval.append(loss.item())
            batch_loss = torch.sum(torch.tensor(running_loss_eval)) / len(val_loader)
        return batch_loss

    def train_pretext(self, train_loader, val_loader, model, optimizer, criterion, epoch, loss_history, val_history):
        """
        Implements the training loop
        :param train_loader: the dataloader used to load the data from the training split
        :param val_loader: the dataloader used to load the data from the validation split
        :param model: the model currently being trained
        :param optimizer: the optimizer used to train the model
        :param criterion: the criterion used to calculate the loss for the pretext task
        :param epoch: the number epochs already executed
        :param loss_history: the history of the training loss
        :param val_history: the history of the validation loss
        :return:
        """
        loop = tqdm(range(epoch, self.n_epochs_pretext), total=self.n_epochs_pretext - epoch, leave=False)
        for e in loop:
            # train
            model.train()
            batch_loss = self.train_batch(train_loader, model, criterion, optimizer)
            loss_history.append(batch_loss.item())
            # validate
            model.eval()
            val_loss = self.validate(val_loader, model, criterion)
            val_history.append(val_loss.item())
            loop.set_description(f'Epoch pretext processed')
            loop.set_postfix(train_loss=loss_history[-1], val_loss=val_history[-1])
            # save checkpoint
            checkpoint = {
                'epoch': e,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_history': loss_history,
                'val_history': val_history
            }
            utility.save_checkpoint(checkpoint,
                                    path=os.path.join(os.curdir, 'saved_models/personal_checkpoint_pretext.pth'))
        # save model
        utility.save_model(model, path=os.path.join(os.curdir, 'saved_models/personal_model_pretext.pth'))
