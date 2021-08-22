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

    def __init__(self, mode='train', T=20, split=[0.9, 0.1]):
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

    def augment_image(self, x, patch_size=14):
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
        batch = self.build_data(self.data[idx])
        return batch

    def __len__(self):
        return len(self.data)
        # return 100


class ContrastiveLearningDataPretext(Dataset):

    imgs_path = 'data/Resized/Unlabelled'

    def __init__(self, mode='train', split=[0.9, 0.1]):
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
        # return 100


class JiGenData(Dataset):

    imgs_path = 'data/Resized/Labelled/Images'
    gt_path = 'data/Resized/Labelled/Groundtruth'

    def __init__(self, P, N=3, split=[0.2, 0.1, 0.7], mode='train'):
        super(JiGenData, self).__init__()
        self.P = P
        self.N = N
        self.mode = mode
        self.imgs_labels = sorted(os.listdir(os.path.join(os.curdir, JiGenData.imgs_path)))
        self.gt_labels = sorted(os.listdir(os.path.join(os.curdir, JiGenData.gt_path)))
        if mode == 'train':
            start_idx = 0
            end_idx = int(len(self.imgs_labels) * split[0])
            self.data = self.get_data(start_idx, end_idx)
        elif mode == 'val':
            start_idx = int(len(self.imgs_labels) * split[0])
            end_idx = start_idx + int(len(self.imgs_labels) * split[1])
            self.data = self.get_data(start_idx, end_idx)
        elif mode == 'test':
            start_idx = int(len(self.imgs_labels) * (split[0] + split[1]))
            end_idx = len(self.imgs_labels)
            self.data = self.get_data(start_idx, end_idx)

    def get_data(self, start_idx, end_idx):
        data = [(img, gt) for img, gt in zip(self.imgs_labels[start_idx:end_idx], self.gt_labels[start_idx:end_idx])]
        return data


    def get_permutation_set(self):
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        permutations = [indices]
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

    def get_segmentation_batch(self, idx):
        img_path = os.path.join(JiGenData.imgs_path, self.imgs_labels[idx])
        gt_path = os.path.join(JiGenData.gt_path, self.gt_labels[idx])
        img = Image.open(os.path.join(os.curdir, img_path)).convert("RGB")
        gt = np.array(Image.open(os.path.join(os.curdir, gt_path)).convert('L'), dtype=np.float32)
        gt[gt > 0] = 1.0
        tensor_converter = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                            (0.5, 0.5, 0.5))])
        gt_tensor_converter = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
        img = tensor_converter(img)
        gt = gt_tensor_converter(gt)
        return img, gt

    def build_data(self, img_label, permutations):
        imgs = torch.ones((self.P, 3, 128, 128))
        labels = torch.ones(self.P, dtype=int)
        for idx in range(len(permutations)):
            imgs[idx], labels[idx] = self.permute_img(img_label, permutations)
        return imgs, labels

    def permute_img(self, img_label, permutations_set):
        # retrieve the image
        # apply a 3x3 grid
        # calculate 100 permutations using the Hamming distance
        # append each permutation into a list and return it
        random_choice = random.randint(0, len(permutations_set)-1)
        chosen_p = permutations_set[random_choice]
        dir_path = os.path.join(os.curdir, JiGenData.imgs_path)
        img_path = os.path.join(dir_path, img_label)
        img = Image.open(img_path)
        permuted_img = self.shuffle_tiles(img, chosen_p)
        return permuted_img, random_choice

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

    def __getitem__(self, idx):
        # each element in a batch is as follows [P_shuffled_imgs, idx_permutations, N_original_imgs, N_gt_imgs]
        seg_img, gt_img = self.get_segmentation_batch(idx)
        permutation_set = self.get_permutation_set()
        shuffled_imgs, perm_idx = self.build_data(self.imgs_labels[idx], permutation_set)
        return shuffled_imgs, perm_idx, seg_img, gt_img

    def __len__(self):
        # return 100
        return len(self.data)


class CustomDataPretext(Dataset):

    unlabelled_path = 'data/Resized/Unlabelled'

    def __init__(self, mode='train', split=[0.9, 0.1]):
        super(CustomDataPretext, self).__init__()
        self.imgs_label = os.listdir(os.path.join(os.curdir, CustomDataPretext.unlabelled_path))
        self.mode = mode
        self.split = split
        self.data = self.get_data()
        self.augmentations = {0: torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                 torchvision.transforms.Normalize((.5, .5, .5),
                                                                                                  (.5, .5, .5))
                                                                 ]),
                              1: torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                 torchvision.transforms.Normalize((.5, .5, .5),
                                                                                                  (.5, .5, .5)),
                                                                 torchvision.transforms.RandomResizedCrop((128, 128))
                                                                 ]),
                              2: torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                 torchvision.transforms.Normalize((.5, .5, .5),
                                                                                                  (.5, .5, .5)),
                                                                 torchvision.transforms.GaussianBlur(5,
                                                                                                     sigma=(1.0, 3.5))
                                                                 ]),
                              3: torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                 torchvision.transforms.Normalize((.5, .5, .5),
                                                                                                  (.5, .5, .5)),
                                                                 torchvision.transforms.ColorJitter(0.9, 0.9, 0.9, 0.5)
                                                                 ])
                              }

    def process_batch(self, idx):
        # image path
        image_label = self.data[idx]
        abs_path = os.path.join(os.curdir, CustomDataPretext.unlabelled_path)
        img_path = os.path.join(abs_path, image_label)
        image = Image.open(os.path.join(os.curdir, img_path))
        augmentation_idx = random.randint(0, len(self.augmentations)-1)
        image = self.augmentations[augmentation_idx](image)
        return image, augmentation_idx

    def __getitem__(self, idx):
        return self.process_batch(idx)

    def __len__(self):
        return len(self.data)
        # return 100

    def get_data(self):
        if self.mode == 'train':
            return self.imgs_label[:int(len(self.imgs_label) * self.split[0])]
        elif self.mode == 'val':
            return self.imgs_label[int(len(self.imgs_label) * self.split[0]):]

    def visualize_image(self, x):
        plt.figure(figsize=(16, 16))
        x = x.permute(1, 2, 0)
        x = (x * 0.5) + 0.5
        plt.imshow(x)
        plt.axis('off')
        plt.show()
        plt.close()


# dataset = JigsawData(P=10)
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# for batch in dataloader:
#     pretext_imgs, labels, seg_imgs, gt_imgs = batch
#     print()

# dataset = CustomDataPretext()
# dataloader = DataLoader(dataset, batch_size=64)
# for batch in dataloader:
#     imgs = batch[0]
#     labels = batch[1]
#     print()
