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
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_jigsaw_pretext(batch_size):
    model = JiGen().to(DEVICE)
    dataset = JigsawDataPretext()
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader = DataLoader(dataset, **dataloader_params)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    return dataloader, model, optimizer, criterion


def get_context_restoration_pretext(batch_size):
    model = ContextRestoration(in_channel=3).to(DEVICE)
    dataset = ContextRestorationDataPretext(T=20)
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader = DataLoader(dataset, **dataloader_params)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    return dataloader, model, optimizer, criterion


def get_contrastive_learning_pretext(batch_size):
    model = SimCLR().to(DEVICE)
    dataset = ContrastiveLearningDataPretext()
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader = DataLoader(dataset, **dataloader_params)
    criterion = ContrastiveLoss().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters())
    return dataloader, model, optimizer, criterion


def get_segmentation(batch_size, technique):
    if technique == 'jigsaw':
        model = JiGen().to(DEVICE)
    elif technique == 'context_restoration':
        model = ContextRestoration(in_channel=3)
    elif technique == 'contrastive_learning':
        model = SimCLR().to(DEVICE)
    dataset = SegmentationDataset(mode='train')
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader = DataLoader(dataset, **dataloader_params)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    return dataloader, model, optimizer, criterion


class Trainer(ABC):

    def __init__(self, n_epochs):
        self.n_epochs = n_epochs

    @abstractmethod
    def train_pretext(self, dataloader, model, optimizer, criterion):
        pass

    def train_segmentation(self, dataloader, model, optimizer, criterion):
        model.train()
        train_history = []
        val_history = []
        # load the model from pretext task
        for e in range(self.n_epochs):
            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
            running_loss = []
            for idx, (imgs, gt_imgs) in loop:
                prediction = model(imgs.to(DEVICE))
                loss = criterion(prediction, gt_imgs.to(DEVICE))
                loop.set_description(f"Batch [{idx}]/[{len(dataloader)}]")
                loop.set_postfix(loss=loss.item())
                running_loss.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print statistics
            batch_loss = torch.sum(torch.tensor(running_loss))/len(dataloader)
            train_history.append(batch_loss.item())
            print(f'Epoch [{e}]/[{self.n_epochs}], average_loss: {batch_loss:.2f}')
            # save checkpoint
            checkpoint = {
                'epoch': e,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_history': train_history,
                'val_history': val_history
            }
            utility.save_checkpoint(checkpoint, path=os.path.join(os.curdir, 'model/segmentation_checkpoint.pth'))
        # save the model
        utility.save_model(model, path=os.path.join(os.curdir, 'model/segmentation_model.pth'))


class JigsawTrainer(Trainer):

    def __init__(self, n_epochs, P, N):
        super(JigsawTrainer, self).__init__(n_epochs)
        self.n_epochs = n_epochs
        self.P = P
        self.N = N

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
        imgs = torch.ones((64, 30, 3, 128, 128))
        labels = torch.ones((64, 30), dtype=int)
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

    def train_pretext(self, dataloader, model, optimizer, criterion):
        model.train()
        permutation_set = self.get_permutation_set()
        train_history = []
        val_history = []
        for e in range(self.n_epochs):
            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
            running_loss = []
            for idx, batch in loop:
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
                running_loss.append(torch.sum(torch.tensor(loss_permutations))/self.P)
                loop.set_description(f"Batch [{idx}]/[{len(dataloader)}]")
                loop.set_postfix(loss=running_loss[-1].item())
            # print statistics
            batch_loss = torch.sum(torch.tensor(running_loss))/len(dataloader)
            train_history.append(batch_loss.item())
            print(f'Epoch [{e}]/[{self.n_epochs}], average_loss: {batch_loss:.2f}')
            # checkpoint
            checkpoint = {
                'epoch': e,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss_history': train_history,
                'val_history': val_history
            }
            utility.save_checkpoint(checkpoint, path=os.path.join(os.curdir, 'model/jigsaw_checkpoint_pretext.pth'))
        # save the model
        utility.save_model(model, path=os.path.join(os.curdir, 'model/jigsaw_model_pretext.pth'))

    def evaluate(self, model, t):
        # to evaluate the model use the Jaccard Index with a threshold t
        pass


class ContrastiveLearningTrainer(Trainer):

    def __init__(self, n_epochs):
        super(ContrastiveLearningTrainer, self).__init__(n_epochs)
        self.n_epochs = n_epochs

    def train_pretext(self, dataloader, model, optimizer, criterion):
        model.train()
        train_history = []
        val_history = []
        for e in range(self.n_epochs):
            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
            running_loss = []
            for idx, x in loop:
                processed_1, processed_2 = model(x.to(DEVICE), pretext=True)
                loss = criterion(processed_1.to(DEVICE), processed_2.to(DEVICE))
                loop.set_description(f"Epoch [{idx}]/[{len(dataloader)}]")
                loop.set_postfix(loss=loss.item())
                running_loss.append(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # print statistics
            batch_loss = torch.sum(torch.tensor(running_loss))/len(dataloader)
            train_history.append(batch_loss.item())
            print(f'Epoch [{e}]/[{self.n_epochs}], average_loss: {batch_loss:.2f}')
            # save checkpoint
            checkpoint = {
                'epoch': e,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_history': train_history,
                'val_history': val_history
            }
            utility.save_checkpoint(checkpoint, path=os.path.join(os.curdir, 'model/contr_lear_checkpoint_pretext.pth'))
        # save model
        utility.save_model(model, path=os.path.join(os.curdir, 'model/contr_lear_model_pretext.pth'))

class ContextRestorationTrainer(Trainer):

    def __init__(self, n_epochs):
        super(ContextRestorationTrainer, self).__init__(n_epochs)
        self.n_epochs = n_epochs

    def train_pretext(self, dataloader, model, optimizer, criterion):
        model.train()
        train_history = []
        val_history = []
        for e in range(self.n_epochs):
            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
            running_loss = []
            for idx, (corrupted, original) in loop:
                restored = model(corrupted.to(DEVICE), pretext=True)
                loss = criterion(restored.to(DEVICE), original.to(DEVICE))
                loop.set_description(f"Batch [{idx}]/[{len(dataloader)}]")
                loop.set_postfix(loss=loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())
                torch.cuda.empty_cache()
            # print statistics
            batch_loss = torch.sum(torch.tensor(running_loss))/len(dataloader)
            train_history.append(batch_loss.item())
            print(f'Epoch [{e}]/[{self.n_epochs}], average_loss: {batch_loss:.2f}')
            checkpoint = {
                'epoch': e,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_history': train_history,
                'val_history': val_history
            }
            # save checkpoint
            utility.save_checkpoint(checkpoint, path=os.path.join(os.curdir, 'model/cxt_res_checkpoint_pretext.pth'))
        # save model
        utility.save_model(model, path=os.path.join(os.curdir, 'model/cxt_res_model_pretext.pth'))

# Jigsaw
# trainer = JigsawTrainer(n_epochs=5, P=30, N=3)
# dataloader_p, model_p, optimizer_p, criterion_p = get_jigsaw_pretext(batch_size=64)
# trainer.train_pretext(dataloader_p, model_p, optimizer_p, criterion_p)
# dataloader_s, model_s, optimizer_s, criterion_s = get_segmentation(batch_size=64, technique='jigsaw')
# trainer.train_segmentation(dataloader_s, model_s, optimizer_s, criterion_s)

# Context Restoration
# trainer = ContextRestorationTrainer(n_epochs=5)
# dataloader_p, model_p, optimizer_p, criterion_p = get_context_restoration_pretext(batch_size=64)
# trainer.train_pretext(dataloader_p, model_p, optimizer_p, criterion_p)
# dataloader_s, model_s, optimizer_s, criterion_s = get_segmentation(batch_size=64, technique='context_restoration')
# trainer.train_segmentation(dataloader_s, model_s, optimizer_s, criterion_s)

# Contrastive Learning
# trainer = ContrastiveLearningTrainer(n_epochs=5)
# dataloader_p, model_p, optimizer_p, criterion_p = get_contrastive_learning_pretext(batch_size=64)
# trainer.train_pretext(dataloader_p, model_p, optimizer_p, criterion_p)
# dataloader_s, model_s, optimizer_s, criterion_s = get_segmentation(batch_size=64, technique='contrastive_learning')
# trainer.train_segmentation(dataloader_s, model_s, optimizer_s, criterion_s)