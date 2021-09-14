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

def get_jigen_data(P, batch_size, full_data):
    model = JiGen(P=P).to(DEVICE)
    train_data = JiGenData(P=P, mode='train', full_data=full_data, split=[0.6, 0.1, 0.3])
    val_data = JiGenData(P=P, mode='val', full_data=full_data, split=[0.6, 0.1, 0.3])
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


def get_supervised_data(batch_size):
    model = UNET(in_channels=3, name='supervised').to(DEVICE)
    train_data = SegmentationDataset(mode='train', split_perc=[0.6, 0.1, 0.3])
    val_data = SegmentationDataset(mode='val', split_perc=[0.6, 0.1, 0.3])
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader_train = DataLoader(train_data, **dataloader_params)
    dataloader_val = DataLoader(val_data, **dataloader_params)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.1)
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


def get_contrastive_learning_pretext(batch_size, full_data):
    model = SimCLR().to(DEVICE)
    train_data = ContrastiveLearningDataPretext(mode='train', full_data=full_data)
    val_data = ContrastiveLearningDataPretext(mode='val', full_data=full_data)
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader_train = DataLoader(train_data, **dataloader_params)
    dataloader_val = DataLoader(val_data, **dataloader_params)
    criterion = ContrastiveLoss().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.01, nesterov=True)
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


def get_context_restoration_pretext(T, batch_size, full_data):
    # model = ContextRestoration(in_channel=3).to(DEVICE)
    model = UNET(in_channels=3).to(DEVICE)
    train_data = ContextRestorationDataPretext(T=T, mode='train', full_data=full_data)
    val_data = ContextRestorationDataPretext(T=T, mode='val', full_data=full_data)
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader_train = DataLoader(train_data, **dataloader_params)
    dataloader_val = DataLoader(val_data, **dataloader_params)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=0.01, lr=0.001, momentum=0.995, nesterov=True)
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


def get_custom_approach_pretext(batch_size, full_data):
    model = CustomSegmentation(n_augmentations=4).to(DEVICE)
    train_data = CustomDataPretext(mode='train', full_data=full_data)
    val_data = CustomDataPretext(mode='val', full_data=full_data)
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader_train = DataLoader(train_data, **dataloader_params)
    dataloader_val = DataLoader(val_data, **dataloader_params)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.01, momentum=0.995, nesterov=True)
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


def get_segmentation(model, optimizer, batch_size, technique):
    train_data = SegmentationDataset(mode='train', split_perc=[0.2, 0.1, 0.7])
    val_data = SegmentationDataset(mode='val', split_perc=[0.2, 0.1, 0.7])
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


def get_eval_dataset():
    dataset = SegmentationDataset(mode='test', split_perc=[0.2, 0.1, 0.7])
    dataloader_params = {'shuffle': True, 'batch_size': 1}
    dataloader = DataLoader(dataset, **dataloader_params)
    return dataloader


class Trainer(ABC):

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

    def __init__(self, n_epochs, P, N, batch_size, technique='jigen', alpha=0.9, beta=0.6):
        self.P = P
        self.N = N
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.technique = technique
        self.alpha = alpha
        self.beta = beta

    def train(self, model, train_loader, val_loader, optimizer, criterion_ptx, criterion_seg, epoch, loss_history, val_history):
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
            for n, (imgs, label) in enumerate(zip(ptx_img, labels)):
                out_ptx = model(imgs, pretext=True)
                pretext_loss = criterion_ptx(out_ptx, label)
                permutation_loss += (self.alpha*pretext_loss)
            permutation_loss = torch.sum(permutation_loss) / self.P
            total_loss = permutation_loss + loss_seg
            batch_loss.append(total_loss)
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        batch_loss = torch.sum(torch.tensor(batch_loss) / len(data_loader))
        return batch_loss

    def validate(self, model, val_loader, criterion_seg, criterion_ptx):
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

    def train_pretext(self, train_loader, val_loader, model, optimizer, criterion, epoch, loss_history, val_history):
        loop = tqdm(range(epoch, self.n_epochs_pretext), total=self.n_epochs_pretext-epoch, leave=False)
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
            # checkpoint
            checkpoint = {
                'epoch': e,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_history': loss_history,
                'val_history': val_history
            }
            utility.save_checkpoint(checkpoint, path=os.path.join(os.curdir,
                                                                  'saved_models/jigen_checkpoint_pretext.pth'))
        # save the model
        utility.save_model(model, path=os.path.join(os.curdir, 'saved_models/jigen_model_pretext.pth'))

    def evaluate(self, dataloader, model, p_threshold, T):
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

    def __init__(self, n_epochs_pretext, n_epochs_segmentation):
        super(ContrastiveLearningTrainer, self).__init__(n_epochs_pretext, n_epochs_segmentation, technique='contrastive')
        self.n_epochs_pretext = n_epochs_pretext
        self.n_epochs_segmentation = n_epochs_segmentation

    def train_batch(self, train_loader, model, criterion, optimizer):
        running_loss = []
        for idx, x in enumerate(train_loader):
            processed_1, processed_2 = model(x.to(DEVICE), pretext=True)
            loss = criterion(processed_1.to(DEVICE), processed_2.to(DEVICE))
            running_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # print statistics
        batch_loss = torch.sum(torch.tensor(running_loss)) / len(train_loader)
        return batch_loss

    def validate(self, val_loader, model, criterion):
        running_loss_eval = []
        with torch.no_grad():
            for idx, x in enumerate(val_loader):
                processed_1, processed_2 = model(x.to(DEVICE), pretext=True)
                loss = criterion(processed_1.to(DEVICE), processed_2.to(DEVICE))
                running_loss_eval.append(loss)
        return torch.sum(torch.tensor(running_loss_eval)) / len(val_loader)

    def train_pretext(self, train_loader, val_loader, model, optimizer, criterion, epoch, loss_history, val_history):
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

    def __init__(self, n_epochs_pretext, n_epochs_segmentation):
        super(ContextRestorationTrainer, self).__init__(n_epochs_pretext, n_epochs_segmentation, technique='restoration')
        self.n_epochs_pretext = n_epochs_pretext
        self.n_epochs_segmentation = n_epochs_segmentation

    def train_batch(self, train_loader, model, criterion, optimizer):
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

    def __init__(self, n_epochs_pretext, n_epochs_segmentation):
        super(CustomApproachTrainer, self).__init__(n_epochs_pretext, n_epochs_segmentation, technique='custom_approach')
        self.n_epochs_pretext = n_epochs_pretext
        self.n_epochs_segmentation = n_epochs_segmentation

    def train_batch(self, train_loader, model, criterion, optimizer):
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


# Supervised approach
# trainer = SupervisedTrainer(n_epochs_pretext=0, n_epochs_segmentation=2, technique='supervised')
# data, phase = get_supervised_data(batch_size=32)
# if phase == 'train':
#     trainer.train_segmentation(**data)
# model = data['model']
# test_data = SegmentationDataset(mode='test', split_perc=[0.2, 0.1, 0.7])
# dataloader_test = DataLoader(test_data, batch_size=1, shuffle=True)
# accuracy_t_IoU, accuracy_IoU, dice_score, precision, recall = trainer.evaluate(dataloader_test, model,
#                                                                                p_threshold=0.5, T=0.65)


# JiGen
# trainer = JigenTrainer(n_epochs=10, P=10, N=3, batch_size=32)
# data, phase = get_jigen_data(P=10, batch_size=32)
# if phase == 'train':
#     trainer.train(**data)
# model = data['model']
# test_data = JiGenData(P=10, mode='test')
# dataloader_test = DataLoader(test_data, batch_size=1, shuffle=True)
# accuracy = trainer.evaluate(dataloader_test, model, p_threshold=0.5, T=0.65)

# # FINETUNE
# if phase == 'fine-tune':
#     trainer.train_segmentation(**data_segmentation)
# # otherwise the training is completed, load the model and evaluate
# model = data_segmentation['model']
# test_data = get_eval_dataset()
# # TEST
# accuracy, dice_score, precision, recall = trainer.evaluate(test_data, model, p_threshold=0.6)

# Context Restoration
# trainer = ContextRestorationTrainer(n_epochs_pretext=5, n_epochs_segmentation=10)
# data_pretext, phase = get_context_restoration_pretext(T=20, batch_size=32)
# if phase == 'pretext':
#     trainer.train_pretext(**data_pretext)
# data_segmentation, phase = get_segmentation(data_pretext['model'], data_pretext['optimizer'],
#                                             batch_size=32, technique='context_restoration')
# if phase == 'fine-tune':
#     trainer.train_segmentation(**data_segmentation)
#
# model = data_segmentation['model']
# test_data = get_eval_dataset()
# # TEST
# accuracy, dice_score, precision, recall = trainer.evaluate(test_data, model, p_threshold=0.5, T=0.65)

# Contrastive Learning
# PRETEXT
# trainer = ContrastiveLearningTrainer(n_epochs_pretext=5, n_epochs_segmentation=10)
# data_pretext, phase = get_contrastive_learning_pretext(batch_size=64)
# if phase == 'pretext':
#     trainer.train_pretext(**data_pretext)
# data_segmentation, phase = get_segmentation(data_pretext['model'], data_pretext['optimizer'],
#                                             batch_size=64, technique='contrastive_learning')
# # FINETUNE
# if phase == 'fine-tune':
#     trainer.train_segmentation(**data_segmentation)
# # # otherwise the training is completed, load the model and evaluate
# model = data_segmentation['model']
# test_data = get_eval_dataset()
# # TEST
# accuracy, dice_score, precision, recall = trainer.evaluate(test_data, model, p_threshold=0.5, T=0.65)
# print(accuracy)

# Custom Approach
# PRETEXT
# trainer = CustomApproachTrainer(n_epochs_pretext=5, n_epochs_segmentation=10)
# data_pretext, phase = get_custom_approach_pretext(batch_size=64)
# if phase == 'pretext':
#     trainer.train_pretext(**data_pretext)
# data_segmentation, phase = get_segmentation(data_pretext['model'], data_pretext['optimizer'],
#                                             batch_size=64, technique='custom_approach')
# # FINETUNE
# if phase == 'fine-tune':
#     trainer.train_segmentation(**data_segmentation)
# model = data_segmentation['model']
# test_data = get_eval_dataset()
# # TEST
# accuracy, dice_score, precision, recall = trainer.evaluate(test_data, model, p_threshold=0.5, T=0.65)
# print(accuracy)
