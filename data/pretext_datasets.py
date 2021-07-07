import os
import torchvision
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np

# # in case of testing
# from model.context_restoration import ContextRestoration


class ContextRestorationDataPretext(Dataset):

    imgs_path = 'Resized/Unlabelled'

    def __init__(self, T):
        super(ContextRestorationDataPretext, self).__init__()
        self.T = T
        self.images = os.listdir(os.path.join(os.curdir, ContextRestorationDataPretext.imgs_path))

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
        original, corrupted = self.build_data(self.images[idx])
        return original, corrupted

    def __len__(self):
        return len(self.images)


class ContrastiveLearningDataPretext(Dataset):

    imgs_path = 'Resized/Unlabelled'

    def __init__(self):
        self.images = os.listdir(os.path.join(os.curdir, ContrastiveLearningDataPretext.imgs_path))

    def build_data(self, image_label):
        abs_path = os.path.join(os.curdir, ContrastiveLearningDataPretext.imgs_path)
        img_path = os.path.join(abs_path, image_label)
        img = Image.open(os.path.join(os.curdir, img_path))
        tensor_converter = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize((.5, .5, .5),
                                                                                            (.5, .5, .5))])
        img = tensor_converter(img)
        augmented_1 = self.augment_image(img)
        augmented_2 = self.augment_image(img)
        return augmented_1, augmented_2

    def augment_image(self, x):
        # Step 1: random crop and resize
        # Step 2: random colour distortion
        # Step 3: random gaussian blur
        # output: a randomly augmented image
        augmenter = torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop((x.shape[1], x.shape[2])),
                                                    torchvision.transforms.ColorJitter((1, 2), 2, 2, 0.5),
                                                    torchvision.transforms.GaussianBlur(3)])
        return augmenter(x)

    def visualize_image(self, x):
        plt.figure(figsize=(16, 16))
        x = x.permute(1, 2, 0)
        x = (x * 0.5) + 0.5
        plt.imshow(x)
        plt.axis('off')
        plt.show()
        plt.close()

    def __getitem__(self, idx):
        aug_1, aug_2 = self.build_data(self.images[idx])
        return aug_1, aug_2

    def __len__(self):
        return len(self.images)


class JigsawDataPretext(Dataset):

    unlabelled_path = 'Resized/Unlabelled'

    def __init__(self, P=30, N=3):
        super(JigsawDataPretext, self).__init__()
        self.images = os.listdir(os.path.join(os.curdir, JigsawDataPretext.unlabelled_path))
        self.P = P
        self.N = N

    def __getitem__(self, idx):
        permutations = self.get_permutations(self.images[idx])
        return permutations

    def get_permutations(self, img_label):
        # retrieve the image
        # apply a 3x3 grid
        # calculate 100 permutations using the Hamming distance
        # append each permutation into a list and return it
        permutations = self.all_permutations()
        dir_path = os.path.join(os.curdir, JigsawDataPretext.unlabelled_path)
        img_path = os.path.join(dir_path, img_label)
        img = Image.open(img_path)
        permuted_images = self.shuffle_tiles(img, permutations)  # returns a list of 'shuffled' images
        # tensor_converter = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
        #                                                    torchvision.transforms.Normalize((.5, .5, .5,),
        #                                                                                     (.5, .5, .5))])
        # img = tensor_converter(img)
        return permuted_images

    def shuffle_tiles(self, img, permutations):
        tiles = [None] * self.N**2
        permuted_images = []
        for i in range(self.N**2):
            tiles[i] = self.get_tile(img, i)
        for idx, p in enumerate(permutations):
            if idx == 0:
                img_data = tiles
            else:
                img_data = [tiles[permutations[idx][t]] for t in range(self.N**2)]
            tensor_converter = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                               torchvision.transforms.Normalize((.5, .5, .5,),
                                                                                                (.5, .5, .5))])
            img_data = [tensor_converter(img) for img in img_data]
            img_data = torch.stack(img_data, 0)
            img = torchvision.utils.make_grid(img_data, self.N, padding=0)
            upsampler = torchvision.transforms.Resize((128, 128))
            img = upsampler(img)
            self.visualize_image(img)
            permuted_images.append(img)
        return permuted_images

    def visualize_image(self, x):
        plt.figure(figsize=(16, 16))
        x = x.permute(1, 2, 0)
        x = (x * 0.5) + 0.5
        plt.imshow(x)
        plt.axis('off')
        plt.show()
        plt.close()

    def get_tile(self, img, i):
        w = int(img.size[0] / self.N)
        y = int(i/self.N)
        x = i % self.N
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        return tile

    def all_permutations(self, min_hamming_dist=2):
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        permutations = []
        permutations.append(indices)
        while len(permutations) != self.P:
            candidate = np.random.permutation(indices)
            if self.validate_permutation(permutations, candidate, min_hamming_dist):
                permutations.append(candidate.tolist())
        return permutations

    def validate_permutation(self, permutations, candidate, min_dist):
        for p in permutations:
            dist = sum(int(char1 != char2) for char1, char2 in zip(p, candidate))
            if dist < min_dist:
                return False
        return True

    def __len__(self):
        return len(self.images)



# dataset = ContextRestorationDataPretext(20)
# dataloader_params = {"shuffle": True, "batch_size":64}
# epochs = 10
# dataloader = DataLoader(dataset, **dataloader_params)
#
# model = ContextRestoration()
# for e in range(epochs):
#     for batch in dataloader:
#         corrupted = batch[0]
#         original = batch[1]
#         output = model(corrupted)

# dataset = JigsawDataPretext()
# permutations = dataset.__getitem__(56)
# print(permutations)

# dataset = ContrastiveLearningDataPretext()
# dataloader = DataLoader(dataset, batch_size=10, shuffle=True)
# for image in dataloader:
#     for aug1, aug2 in zip(image[0], image[1]):
#         dataset.visualize_image(aug1)
#         dataset.visualize_image(aug2)

# dataset = ContextRestorationDataPretext(15)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# for image in dataloader:
#     for original, corrupted in zip(image[0], image[1]):
#         dataset.visualize_image(original)
#         dataset.visualize_image(corrupted)
