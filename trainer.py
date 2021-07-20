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


def get_jigsaw_pretext(batch_size):
    model = JiGen()
    dataset = JigsawDataPretext()
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader = DataLoader(dataset, **dataloader_params)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    return dataloader, model, optimizer, criterion


def get_context_restoration_pretext(batch_size):
    model = ContextRestoration(in_channel=3, out_channel=3)
    dataset = ContextRestorationDataPretext(T=20)
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader = DataLoader(dataset, **dataloader_params)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    return dataloader, model, optimizer, criterion


def get_contrastive_learning_pretext(batch_size):
    model = SimCLR()
    dataset = ContrastiveLearningDataPretext()
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader = DataLoader(dataset, **dataloader_params)
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters())
    return dataloader, model, optimizer, criterion


def get_segmentation(batch_size, technique):
    if technique == 'jigsaw':
        model = JiGen()
    elif technique == 'context_restoration':
        model = ContextRestoration(in_channel=3, out_channel=1)
    elif technique == 'contrastive_learning':
        model = SimCLR()
    dataset = SegmentationDataset(mode='train')
    dataloader_params = {'shuffle': True, 'batch_size': batch_size}
    dataloader = DataLoader(dataset, **dataloader_params)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    return dataloader, model, optimizer, criterion


class Trainer(ABC):

    def __init__(self, n_epochs):
        self.n_epochs = n_epochs

    def save_checkpoint(self):
        pass

    def save_model(self):
        pass

    @abstractmethod
    def train_pretext(self, dataloader, model, optimizer, criterion):
        pass

    def train_segmentation(self, dataloader, model, optimizer, criterion):
        model.train()
        # load the model from pretext task
        for e in range(self.n_epochs):
            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
            for idx, (imgs, gt_imgs) in loop:
                prediction = model(imgs)
                loss = criterion(prediction, gt_imgs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # save checkpoint
            # print statistics
            loop.set_description(f"Epoch [{e}]/[{self.n_epochs}]")
            loop.set_postfix(loss=loss.item())
        # save the model


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
        for e in range(self.n_epochs):
            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
            for idx, batch in loop:
                img_batch, labels = self.get_data(batch, permutation_set)
                img_batch = img_batch.permute((1, 0, 2, 3, 4))
                labels = labels.permute((1, 0))
                for imgs, label in zip(img_batch, labels):
                    output = model(imgs, pretext=True)
                    loss = criterion(output, label)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            # save the model's parameter
            # print statistics
            loop.set_description(f"Epoch [{e}]/[{self.n_epochs}]")
            loop.set_postfix(loss=loss.item())
        # save the model

    def evaluate(self, model, t):
        # to evaluate the model use the Jaccard Index with a threshold t
        pass


class ContrastiveLearningTrainer(Trainer):

    def __init__(self, n_epochs):
        super(ContrastiveLearningTrainer, self).__init__(n_epochs)
        self.n_epochs = n_epochs

    def train_pretext(self, dataloader, model, optimizer, criterion):
        model.train()
        for e in range(self.n_epochs):
            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
            for idx, x in loop:
                aug_1, aug_2 = model(x, pretext=True)
                loss = criterion(aug_1, aug_2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # save checkpoint
            # print statistics
            loop.set_description(f"Epoch [{e}]/[{self.n_epochs}]")
            loop.set_postfix(loss=loss.item())
        # save model


class ContextRestorationTrainer(Trainer):

    def __init__(self, n_epochs):
        super(ContextRestorationTrainer, self).__init__(n_epochs)
        self.n_epochs = n_epochs

    def train_pretext(self, dataloader, model, optimizer, criterion):
        model.train()
        for e in range(self.n_epochs):
            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
            for idx, (corrupted, original) in loop:
                restored = model(corrupted, pretext=True)
                loss = criterion(restored, original)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # save checkpoint
            # print statistics
            loop.set_description(f"Epoch [{e}]/[{self.n_epochs}]")
            loop.set_postfix(loss=loss.item())
        # save model


# Jigsaw
# trainer = JigsawTrainer(n_epochs=5, P=30, N=3)
# dataloader_p, model_p, optimizer_p, criterion_p = get_jigsaw_pretext(batch_size=64)
# trainer.train_pretext(dataloader_p, model_p, optimizer_p, criterion_p)
# dataloader_s, model_s, optimizer_s, criterion_s = get_jigsaw_segmentation(batch_size=64, technique='jigsaw')
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