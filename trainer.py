import torch.nn
from data.pretext_datasets import ContextRestorationDataPretext, ContrastiveLearningDataPretext, JigsawDataPretext
from torch.utils.data import DataLoader
from data.segmentation_dataset import SegmentationDataset
from model.context_restoration import ContextRestoration
from model.contrastive_learning import SimCLR
from model.jigsaw import JiGen
from abc import ABC, abstractmethod
from tqdm import tqdm


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

    def __init__(self, n_epochs):
        super(JigsawTrainer, self).__init__(n_epochs)
        self.n_epochs = n_epochs

    def train_pretext(self, dataloader, model, optimizer, criterion):
        model.train()
        for e in range(self.n_epochs):
            loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
            for idx, (imgs, labels) in loop:
                output = model(imgs, pretext=True)
                loss = criterion(output, labels)
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
            for idx, (aug1, aug2) in loop:
                processed_1 = model(aug1)
                processed_2 = model(aug2)
                loss = criterion(processed_1, processed_2)
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
                restored = model(corrupted)
                loss = criterion(restored, original)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # save checkpoint
            # print statistics
            loop.set_description(f"Epoch [{e}]/[{self.n_epochs}]")
            loop.set_postfix(loss=loss.item())
        # save model


model = JiGen()
dataset = JigsawDataPretext()
dataloader_params = {'shuffle': True, 'batch_size': 64}
dataloader = DataLoader(dataset, **dataloader_params)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

trainer = JigsawTrainer(5)
trainer.train_pretext(dataloader, model, optimizer, criterion)

