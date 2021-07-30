import torch.nn
import torch
from data.pretext_datasets import ContextRestorationDataPretext, ContrastiveLearningDataPretext, JigsawDataPretext
from torch.utils.data import DataLoader
from data.segmentation_dataset import SegmentationDataset
from model.context_restoration import ContextRestoration
from model.contrastive_learning import SimCLR
from model.jigsaw import JiGen
from model.utils.criterions import ContrastiveLoss
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


def get_jigsaw_pretext(batch_size, split=[0.8, 0.2]):
    model = JiGen().to(DEVICE)
    train_data = JigsawDataPretext(mode='train')
    val_data = JigsawDataPretext(mode='val')
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader_train = DataLoader(train_data, **dataloader_params)
    dataloader_val = DataLoader(val_data, **dataloader_params)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    epoch = 0
    loss_history = []
    val_history = []
    phase = 'pretext'
    # check if the model has already been trained
    if os.path.exists(os.path.join(saved_files_path, 'jigsaw_model_pretext.pth')):
        model_path = os.path.join(saved_files_path, 'jigsaw_model_pretext.pth')
        model = load_model(model, path=model_path)
        phase = 'segmentation'
    elif os.path.exists(os.path.join(saved_files_path, 'jigsaw_checkpoint_pretext.pth')):
        model_path = os.path.join(saved_files_path, 'jigsaw_checkpoint_pretext.pth')
        epoch, model, optimizer, loss_history, val_history = load_checkpoint(model, optimizer, model_path)
    data = {'model': model, 'optimizer': optimizer,
            'train_loader': dataloader_train, 'val_loader': dataloader_val,
            'criterion': criterion, 'epoch': epoch,
            'loss_history': loss_history, 'val_history': val_history}
    return data, phase


def get_contrastive_learning_pretext(batch_size):
    model = SimCLR().to(DEVICE)
    train_data = ContrastiveLearningDataPretext(mode='train')
    val_data = ContrastiveLearningDataPretext(mode='val')
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader_train = DataLoader(train_data, **dataloader_params)
    dataloader_val = DataLoader(val_data, **dataloader_params)
    criterion = ContrastiveLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
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


def get_context_restoration_pretext(batch_size):
    model = ContextRestoration(in_channel=3).to(DEVICE)
    train_data = ContextRestorationDataPretext(mode='train')
    val_data = ContextRestorationDataPretext(mode='val')
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader_train = DataLoader(train_data, **dataloader_params)
    dataloader_val = DataLoader(val_data, **dataloader_params)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    epoch = 0
    loss_history = []
    val_history = []
    phase = 'pretext'

    if os.path.exists(os.path.join(saved_files_path, 'context_model_pretext.pth')):
        model_path = os.path.join(saved_files_path, 'context_model_pretext.pth')
        model = load_model(model, path=model_path)
        phase = 'segmentation'
    elif os.path.exists(os.path.join(saved_files_path, 'context_checkpoint_pretext.pth')):
        model_path = os.path.join(saved_files_path, 'context_checkpoint_pretext.pth')
        epoch, model, optimizer, loss_history, val_history = load_checkpoint(model, optimizer, model_path)

    data = {'model': model, 'optimizer': optimizer,
            'train_loader': dataloader_train, 'val_loader': dataloader_val,
            'criterion': criterion, 'epoch': epoch,
            'loss_history': loss_history, 'val_history': val_history}
    return data, phase


def get_segmentation(model, optimizer, batch_size, technique):
    train_data = SegmentationDataset(mode='train')
    val_data = SegmentationDataset(mode='val')
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    train_loader = DataLoader(train_data, **dataloader_params)
    val_loader = DataLoader(val_data, **dataloader_params)
    criterion = torch.nn.BCEWithLogitsLoss()
    epoch = 0
    loss_history = []
    val_history = []
    phase = 'fine_tune'
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

    data = {'model': model, 'optimizer': optimizer,
            'train_loader': train_loader, 'val_loader': val_loader,
            'criterion': criterion, 'epoch': epoch,
            'loss_history': loss_history, 'val_history': val_history}
    return data, phase

def get_eval_dataset():
    dataset = SegmentationDataset('test')
    dataloader_params = {'shuffle': True, 'batch_size': 1}
    dataloader = DataLoader(dataset, **dataloader_params)
    return dataloader


class Trainer(ABC):

    def __init__(self, n_epochs, technique):
        self.n_epochs = n_epochs
        self.technique = technique

    @abstractmethod
    def train_pretext(self, dataloader, model, optimizer, criterion):
        pass

    def train_segmentation(self, train_loader, val_loader, model, optimizer, criterion, epoch, loss_history, val_history):
        loop = tqdm(range(epoch, self.n_epochs), total=self.n_epochs-epoch, leave=False)
        # load the model from pretext task
        for e in loop:
            model.train()
            running_loss_train = []
            running_loss_val = []
            # train
            for idx, (imgs, gt_imgs) in enumerate(train_loader):
                prediction = model(imgs.to(DEVICE))
                loss = criterion(prediction, gt_imgs.to(DEVICE))
                running_loss_train.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print statistics
            batch_loss = torch.sum(torch.tensor(running_loss_train))/len(train_loader)
            loss_history.append(batch_loss.item())

            # validation
            for idx, (imgs, gt_imgs) in enumerate(val_loader):
                prediction = model(imgs.to(DEVICE))
                loss = criterion(prediction, gt_imgs.to(DEVICE))
                running_loss_val.append(loss)
            batch_loss = torch.sum(torch.tensor(running_loss_val))/len(val_loader)
            val_history.append(batch_loss.item())
            loop.set_description('Epoch processed')
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

    def evaluate(self, dataloader, model, p_threshold, T=0.3):
        output_path = os.path.join(os.curdir, f'saved_models/predictions_{model.name}')
        prediction_samples = []
        model.eval()
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        ious = []
        with torch.no_grad():
            for idx, batch in loop:
                img = batch[0].to(DEVICE)
                # gt = batch[1].squeeze(1).view(-1).to(DEVICE)
                gt = batch[1].to(DEVICE)
                output = torch.sigmoid(model(img)).squeeze(0)
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
                false_positives = torch.sum(torch.tensor([1.0 if p == 1.0 and g == 0.0 else 0.0
                                                         for p, g in zip(prediction, gt)]))
                false_negatives = torch.sum(torch.tensor([1.0 if p == 0.0 and g == 1.0 else 0.0
                                                         for p, g in zip(prediction, gt)]))
                iou = true_positives / (true_positives + false_negatives + false_positives)
                ious.append((iou > T).float())
                loop.set_postfix(accuracy=torch.sum(torch.tensor(ious)).item()/len(ious))
            accuracy = torch.sum(torch.tensor(ious)).item()/len(ious)
        return accuracy

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


class JigsawTrainer(Trainer):

    def __init__(self, n_epochs, P, N, batch_size):
        super(JigsawTrainer, self).__init__(n_epochs, technique='jigsaw')
        self.n_epochs = n_epochs
        self.P = P
        self.N = N
        self.batch_size = batch_size

    def get_permutation_set(self):
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        permutations = []
        permutations.append(indices)
        while len(permutations) != self.P:
            candidate = np.random.permutation(indices)
            if self.validate_permutation(permutations, candidate, min_dist=4):
                permutations.append(candidate.tolist())
        return permutations

    def validate_permutation(self, permutations, candidate, min_dist=4):
        for p in permutations:
            dist = sum(int(char1 != char2) for char1, char2 in zip(p, candidate))
            if dist < min_dist:
                return False
        return True

    def get_data(self, img_labels, permutations):
        imgs = torch.ones((self.batch_size, self.P, 3, 128, 128))
        labels = torch.ones((self.batch_size, self.P), dtype=int)
        for idx, img_label in enumerate(img_labels):
            for _ in range(len(permutations)):
                imgs[idx], labels[idx] = self.permute_img(img_label, permutations)
        return imgs, labels

    def shuffle_tiles(self, img, chosen_p):
        tiles = [None] * self.N**2
        # chosen_permutations = []
        # permuted_images = []
        for i in range(self.N**2):
            tiles[i] = self.get_tile(img, i)
        img_data = [tiles[chosen_p[t]] for t in range(self.N**2)]
        tensor_converter = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                               torchvision.transforms.Normalize((.5, .5, .5,),
                                                                                                (.5, .5, .5))])
        img_data = [tensor_converter(img) for img in img_data]
        img_data = torch.stack(img_data, 0)
        img = torchvision.utils.make_grid(img_data, self.N, padding=0)
        upsampler = torchvision.transforms.Resize((128, 128))
        img = upsampler(img)
        # self.visualize_image(img)
        return img

    def get_tile(self, img, i):
        w = int(img.size[0] / self.N)
        y = int(i/self.N)
        x = i % self.N
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        return tile

    def permute_img(self, img_label, permutations_set):
        # retrieve the image
        # apply a 3x3 grid
        # calculate 100 permutations using the Hamming distance
        # append each permutation into a list and return it
        random_choice = random.randint(0, len(permutations_set)-1)
        chosen_p = permutations_set[random_choice]
        dir_path = os.path.join(os.curdir, JigsawDataPretext.unlabelled_path)
        img_path = os.path.join(dir_path, img_label)
        img = Image.open(img_path)
        permuted_img = self.shuffle_tiles(img, chosen_p)
        return permuted_img, random_choice

    def visualize_image(self, x):
        plt.figure(figsize=(16, 16))
        x = x.permute(1, 2, 0)
        x = (x * 0.5) + 0.5
        plt.imshow(x)
        plt.axis('off')
        plt.show()
        plt.close()

    def train_batch(self, train_loader, permutation_set, model, criterion, optimizer):
        running_loss_train = []
        for idx, batch in enumerate(train_loader):
            img_batch, labels = self.get_data(batch, permutation_set)
            img_batch = img_batch.permute((1, 0, 2, 3, 4)).to(DEVICE)
            labels = labels.permute((1, 0)).to(DEVICE)
            loss_permutations = []
            for n, (imgs, label) in enumerate(zip(img_batch, labels)):
                output = model(imgs, pretext=True)
                loss = criterion(output, label)
                loss_permutations.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            running_loss_train.append(torch.sum(torch.tensor(loss_permutations)) / self.P)
        # update train statistics
        batch_loss = torch.sum(torch.tensor(running_loss_train)) / len(train_loader)
        return batch_loss

    def validate(self, val_loader, permutation_set, model, criterion):
        running_loss_eval = []
        for idx, batch in enumerate(val_loader):
            img_batch, labels = self.get_data(batch, permutation_set)
            img_batch = img_batch.permute((1, 0, 2, 3, 4)).to(DEVICE)
            labels = labels.permute((1, 0)).to(DEVICE)
            loss_permutations = []
            for n, (imgs, label) in enumerate(zip(img_batch, labels)):
                output = model(imgs, pretext=True)
                loss = criterion(output, label)
                loss_permutations.append(loss.item())
            running_loss_eval.append(torch.sum(torch.tensor(loss_permutations)) / self.P)
        batch_loss = torch.sum(torch.tensor(running_loss_eval)) / len(val_loader)
        return batch_loss

    def train_pretext(self, train_loader, val_loader, model, optimizer, criterion, epoch, loss_history, val_history):
        permutation_set = self.get_permutation_set()
        loop = tqdm(range(epoch, self.n_epochs), total=self.n_epochs-epoch, leave=False)
        for e in loop:
            model.train()
            # train loop
            batch_loss = self.train_batch(train_loader, permutation_set, model, criterion, optimizer)
            loss_history.append(batch_loss.item())
            # val loop
            model.eval()
            val_loss = self.validate(val_loader, permutation_set, model, criterion)
            val_history.append(val_loss.item())
            # print statistics
            loop.set_description(f'Epoch processed')
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
                                                                  'saved_models/jigsaw_checkpoint_pretext.pth'))
        # save the model
        utility.save_model(model, path=os.path.join(os.curdir, 'saved_models/jigsaw_model_pretext.pth'))


class ContrastiveLearningTrainer(Trainer):

    def __init__(self, n_epochs):
        super(ContrastiveLearningTrainer, self).__init__(n_epochs, technique='contrastive')
        self.n_epochs = n_epochs

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
        for idx, x in enumerate(val_loader):
            processed_1, processed_2 = model(x.to(DEVICE), pretext=True)
            loss = criterion(processed_1.to(DEVICE), processed_2.to(DEVICE))
            running_loss_eval.append(loss)
        return torch.sum(torch.tensor(running_loss_eval)) / len(val_loader)

    def train_pretext(self, train_loader, val_loader, model, optimizer, criterion, epoch, loss_history, val_history):
        loop = tqdm(range(epoch, self.n_epochs), total=self.n_epochs - epoch, leave=False)
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
            loop.set_description(f'Epoch processed')
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

    def __init__(self, n_epochs):
        super(ContextRestorationTrainer, self).__init__(n_epochs, technique='restoration')
        self.n_epochs = n_epochs

    def train_batch(self, train_loader, model, criterion, optimizer):
        running_loss = []
        for idx, (corrupted, original) in enumerate(train_loader):
            restored = model(corrupted.to(DEVICE), pretext=True)
            loss = criterion(restored.to(DEVICE), original.to(DEVICE))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss.append(loss.item())
            torch.cuda.empty_cache()
        batch_loss = torch.sum(torch.tensor(running_loss)) / len(train_loader)
        return batch_loss

    def validate(self, val_loader, model, criterion):
        running_loss_eval = []
        for idx, (corrupted, original) in enumerate(val_loader):
            restored = model(corrupted.to(DEVICE), pretext=True)
            loss = criterion(restored.to(DEVICE), original.to(DEVICE))
            running_loss_eval.append(loss.item())
            torch.cuda.empty_cache()
        batch_loss = torch.sum(torch.tensor(running_loss_eval)) / len(val_loader)
        return batch_loss

    def train_pretext(self, train_loader, val_loader, model, optimizer, criterion, epoch, loss_history, val_history):
        loop = tqdm(range(epoch, self.n_epochs), total=self.n_epochs - epoch, leave=False)
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
            loop.set_description(f'Epoch processed')
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


# Jigsaw
# PRETEXT
# trainer = JigsawTrainer(n_epochs=10, P=5, N=3, batch_size=32)
# data_pretext, phase = get_jigsaw_pretext(batch_size=32)
# if phase == 'pretext':
#     trainer.train_pretext(**data_pretext)
# data_segmentation, phase = get_segmentation(data_pretext['model'], data_pretext['optimizer'],
#                                             batch_size=32, technique='jigsaw')
# FINETUNE
# if phase == 'fine_tune':
#     trainer.train_segmentation(**data_segmentation)
# # otherwise the training is completed, load the model and evaluate
# model = data_segmentation['model']
# test_data = get_eval_dataset()
# TEST
# accuracy = trainer.evaluate(test_data, model, p_threshold=0.6)

# Context Restoration
# trainer = ContextRestorationTrainer(n_epochs=5)
# data_pretext, phase = get_context_restoration_pretext(batch_size=32)
# if phase == 'pretext':
#     trainer.train_pretext(**data_pretext)
# data_segmentation, phase = get_segmentation(data_pretext['model'], data_pretext['optimizer'],
#                                             batch_size=32, technique='context_restoration')
# if phase == 'fine_tune':
#     trainer.train_segmentation(**data_segmentation)
#
# model = data_segmentation['model']
# test_data = get_eval_dataset()
# # TEST
# accuracy = trainer.evaluate(test_data, model, p_threshold=0.6)
# Contrastive Learning
# PRETEXT
# trainer = ContrastiveLearningTrainer(n_epochs=5)
# data_pretext, phase = get_contrastive_learning_pretext(batch_size=32)
# if phase == 'pretext':
#     trainer.train_pretext(**data_pretext)
# data_segmentation, phase = get_segmentation(data_pretext['model'], data_pretext['optimizer'],
#                                             batch_size=32, technique='contrastive_learning')
# # FINETUNE
# if phase == 'fine_tune':
#     trainer.train_segmentation(**data_segmentation)
# # # otherwise the training is completed, load the model and evaluate
# model = data_segmentation['model']
# test_data = get_eval_dataset()
# # TEST
# accuracy = trainer.evaluate(test_data, model, p_threshold=0.6)
# print(accuracy)