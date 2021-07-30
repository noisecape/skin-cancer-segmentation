import os
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np
import itertools
from torch.random import default_generator
# # in case of testing
# from model.context_restoration import ContextRestoration
from tqdm import tqdm


class ContextRestorationDataPretext(Dataset):

    imgs_path = 'data/Resized/Unlabelled'

    def __init__(self, mode='train', T=20, split=[0.8, 0.2]):
        super(ContextRestorationDataPretext, self).__init__()
        self.T = T
        self.images = os.listdir(os.path.join(os.curdir, ContextRestorationDataPretext.imgs_path))
        self.mode = mode
        self.split = split
        self.data = self.get_data()

    def get_data(self):
        if self.mode == 'train':
            return self.images[:int(len(self.images) * self.split[0])]
        elif self.mode == 'val':
            return self.images[int(len(self.images) * self.split[0]):]

    def build_data(self, img_label):
        img_path = os.path.join(ContextRestorationDataPretext.imgs_path, img_label)
        img = Image.open(os.path.join(os.curdir, img_path))
        tensor_converter = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize((.5, .5, .5),
                                                                                            (.5, .5, .5))])
        img = tensor_converter(img)
        corrupted_img = self.augment_image(img)
        return img, corrupted_img

    def augment_image(self, x, patch_size=10):
        new_x = x.clone().detach()
        for _ in range(self.T):
            firstpatch_coordinates = (random.randint(0, x.shape[1]-patch_size),
                                      random.randint(0, x.shape[2]-patch_size))

            secondpatch_coordinates = (random.randint(0, x.shape[1]-patch_size),
                                       random.randint(0, x.shape[2]-patch_size))

            new_x = self.swap_patches(new_x, firstpatch_coordinates, secondpatch_coordinates, patch_size)

        return new_x

    def swap_patches(self, img, coord_1, coord_2, size):
        # iterate through all the image's channel and swap the selected pixels
        new_x = img.clone().detach()
        first_path = img[0:3, coord_2[0]:coord_2[0]+size, coord_2[1]:coord_2[1]+size]
        second_patch = img[0:3, coord_1[0]:coord_1[0]+size, coord_1[1]:coord_1[1]+size]
        new_x[0:3, coord_1[0]:coord_1[0]+size, coord_1[1]:coord_1[1]+size] = first_path
        new_x[0:3, coord_2[0]:coord_2[0]+size, coord_2[1]:coord_2[1]+size] = second_patch
        return new_x

    def visualize_sample_image(self, n_images=4):
        """
        This function visualize n_images, sampled randomly
        from the dataset
        :param n_images: number of image to print per axis
        :return:
        """
        images = random.sample(self.data, n_images**2)
        plt.figure(figsize=(16, 16))
        for idx, img in enumerate(images):
            plt.subplot(n_images, n_images, idx + 1)
            image = image.permute(1, 2, 0)
            image = (image * 0.5) + 0.5
            plt.imshow(image)
            plt.axis('off')
        plt.show()
        plt.close()

    def visualize_image(self, x):
        plt.figure(figsize=(16, 16))
        x = x.permute(1, 2, 0)
        x = (x * 0.5) + 0.5
        plt.imshow(x)
        plt.axis('off')
        plt.show()
        plt.close()

    def __getitem__(self, idx):
        original, corrupted = self.build_data(self.data[idx])
        return original, corrupted

    def __len__(self):
        return len(self.data)


class ContrastiveLearningDataPretext(Dataset):

    imgs_path = 'data/Resized/Unlabelled'

    def __init__(self, mode='train', split=[0.8, 0.2]):
        super(ContrastiveLearningDataPretext, self).__init__()
        self.images = os.listdir(os.path.join(os.curdir, ContrastiveLearningDataPretext.imgs_path))
        self.mode = mode
        self.split = split
        self.data = self.get_data()

    def get_data(self):
        if self.mode == 'train':
            return self.images[:int(len(self.images) * self.split[0])]
        elif self.mode == 'val':
            return self.images[int(len(self.images) * self.split[0]):]

    def visualize_image(self, x):
        plt.figure(figsize=(16, 16))
        x = x.permute(1, 2, 0)
        x = (x * 0.5) + 0.5
        plt.imshow(x)
        plt.axis('off')
        plt.show()
        plt.close()

    def __getitem__(self, idx):
        x = self.data[idx]
        abs_path = os.path.join(os.curdir, ContrastiveLearningDataPretext.imgs_path)
        img_path = os.path.join(abs_path, x)
        img = Image.open(os.path.join(os.curdir, img_path))
        tensor_converter = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize((.5, .5, .5),
                                                                                            (.5, .5, .5))])
        img = tensor_converter(img)
        return img

    def __len__(self):
        return len(self.data)


class JigsawDataPretext(Dataset):

    unlabelled_path = 'data/Resized/Unlabelled'

    def __init__(self, mode='train', split=[0.8, 0.2]):
        super(JigsawDataPretext, self).__init__()
        self.imgs_label = os.listdir(os.path.join(os.curdir, JigsawDataPretext.unlabelled_path))
        self.mode = mode
        self.split = split
        self.data = self.get_data()

    def get_data(self):
        # The split involves only the training and the validation as the testing phase
        # is done during fine-tune, on a different set of images.
        if self.mode == 'train':
            return self.imgs_label[:int(len(self.imgs_label)*self.split[0])]
        elif self.mode == 'val':
            return self.imgs_label[int(len(self.imgs_label) * self.split[0]):]

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def visualize_image(self, x):
        plt.figure(figsize=(16, 16))
        x = x.permute(1, 2, 0)
        x = (x * 0.5) + 0.5
        plt.imshow(x)
        plt.axis('off')
        plt.show()
        plt.close()

