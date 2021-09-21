import os
import torchvision
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import matplotlib.pyplot as plt
import numpy as np


class ContextRestorationDataPretext(Dataset):
    """
    This class represents the custom dataset for
    the pretext task of the Context Restoration model.
    """
    def __init__(self, mode='train', T=20, split=[0.9, 0.1], full_data=False):
        super(ContextRestorationDataPretext, self).__init__()
        self.full_data = full_data
        if self.full_data:
            self.imgs_path = 'data/Resized_All/Unlabelled'
        else:
            self.imgs_path = 'data/Resized/Unlabelled'
        self.T = T
        self.images = os.listdir(os.path.join(os.curdir, self.imgs_path))
        self.mode = mode
        self.split = split
        self.data = self.get_data()

    def get_data(self):
        """
        Select the path of each image
        according to the split percentage.
        :return:
        """
        if self.mode == 'train':
            return self.images[:int(len(self.images) * self.split[0])]
        elif self.mode == 'val':
            return self.images[int(len(self.images) * self.split[0]):]

    def build_data(self, img_label):
        """
        Each image path gets processed so that the path (string)
        is converted to a tensor, where each value represents the
        value of the pixels. The image is then corrupted by swapping the
        content of two pairs of patches.
        :param img_label: the path of a specific image in the collection
        :return (img, corrupted_img): img is the original image, corrupted_img is the corrupted one.
        """
        img_path = os.path.join(self.imgs_path, img_label)
        img = Image.open(os.path.join(os.curdir, img_path))
        tensor_converter = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                           torchvision.transforms.Normalize((.5, .5, .5),
                                                                                            (.5, .5, .5))])
        img = tensor_converter(img)
        corrupted_img = self.augment_image(img)
        return img, corrupted_img

    def augment_image(self, x, patch_size=14):
        """
        Augment the image by swapping the content of two patches.
        :param x: the image to augment
        :param patch_size: the size of each patch
        :return new_x: the new image where two patches have been swapped.
        """
        new_x = x.clone().detach()
        for _ in range(self.T):
            firstpatch_coordinates = (random.randint(0, x.shape[1]-patch_size),
                                      random.randint(0, x.shape[2]-patch_size))

            secondpatch_coordinates = (random.randint(0, x.shape[1]-patch_size),
                                       random.randint(0, x.shape[2]-patch_size))

            new_x = self.swap_patches(new_x, firstpatch_coordinates, secondpatch_coordinates, patch_size)

        return new_x

    def swap_patches(self, img, coord_1, coord_2, size):
        """
        Perform the actual swapping of the patches.
        :param img: the image whose pair of patches have to be swapped.
        :param coord_1: the coordinates of the first patch in (x, y)
        :param coord_2: the coordinates of the second patch in (x, y)
        :param size: the size of the patches
        :return:
        """
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
        """
        Return a specific pair of (original, corrupted) image.
        :param idx: The index used to access the specific image in the collection.
        :return batch: a tuple containing the original image and the corrupted one
        """
        batch = self.build_data(self.data[idx])
        return batch

    def __len__(self):
        return len(self.data)
        # return 100


class ContrastiveLearningDataPretext(Dataset):
    """
    This class represents the custom dataset for
    the pretext task of SimCLR.
    """

    def __init__(self, mode='train', split=[0.9, 0.1], full_data=False):
        super(ContrastiveLearningDataPretext, self).__init__()
        self.full_data = full_data
        if self.full_data:
            self.imgs_path = 'data/Resized_All/Unlabelled'
        else:
            self.imgs_path = 'data/Resized/Unlabelled'
        self.images = os.listdir(os.path.join(os.curdir, self.imgs_path))
        self.mode = mode
        self.split = split
        self.data = self.get_data()

    def get_data(self):
        """
        Perform the split of the data according to the given percentages
        :return: the set of images path whose cardinality depends by the split percentage
        """
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
        """
        Returns a tensor representing an image selected from the list of path
        defined previously.
        :param idx: the index used to access the idx-th image in the collection.
        :return img: the tensor representing the idx-th image from the collection
        """
        x = self.data[idx]
        abs_path = os.path.join(os.curdir, self.imgs_path)
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
    """
    This class represents the custom dataset used
    for JiGen.
    """

    def __init__(self, P, N=3, split=[0.2, 0.1, 0.7], mode='train', full_data=False):
        super(JiGenData, self).__init__()
        self.full_data = full_data
        if self.full_data:
            self.imgs_path = 'data/Resized_All/Labelled/Images'
            self.gt_path = 'data/Resized_All/Labelled/Groundtruth'
        else:
            self.imgs_path = 'data/Resized/Labelled/Images'
            self.gt_path = 'data/Resized/Labelled/Groundtruth'
        self.P = P
        self.N = N
        self.mode = mode
        self.imgs_labels = sorted(os.listdir(os.path.join(os.curdir, self.imgs_path)))
        self.gt_labels = sorted(os.listdir(os.path.join(os.curdir, self.gt_path)))
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
        """
        Returns a list of pairs of images path from the collection of images,
        where the first image is the jigsaw puzzle while the second image is the original one.
        :param start_idx: the start index defined by the split percentage
        :param end_idx: the end index defined by the plit percentage
        :return:
        """
        data = [(img, gt) for img, gt in zip(self.imgs_labels[start_idx:end_idx], self.gt_labels[start_idx:end_idx])]
        return data

    def get_permutation_set(self):
        """
        Calculate the permutations required to define the Jigsaw puzzles
        for a batch of images
        :return permutations: a list of permutations
        """
        indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        permutations = [indices]
        while len(permutations) != self.P:
            candidate = np.random.permutation(indices)
            if self.validate_permutation(permutations, candidate, min_dist=4):
                permutations.append(candidate.tolist())
        return permutations

    def validate_permutation(self, permutations, candidate, min_dist=4):
        """
        This function validate a specific permutation applying the Hamming distance
        algorithm
        :param permutations: the set of permutations already validated
        :param candidate: the candidate permutation to be validated
        :param min_dist: the minimum number of elements that must be different
        between the candidate and the set of permutations
        :return boolean: True if the permutation is valid, False otherwise.
        """
        for p in permutations:
            dist = sum(int(char1 != char2) for char1, char2 in zip(p, candidate))
            if dist < min_dist:
                return False
        return True

    def get_segmentation_batch(self, idx):
        """
        Builds a pair of original image and the corresponding segmentation mask. The segmentation
        mask is processed so that each pixel either have a value 0 or 1. The original image
        is converted to a tensor.
        :param idx: the index use to retrieve the idx-th image from the collection.
        :return (img, gt): img is the idx-th image in the collection, gt is the corresponding segmentation mask.
        """
        img_path = os.path.join(self.imgs_path, self.imgs_labels[idx])
        gt_path = os.path.join(self.gt_path, self.gt_labels[idx])
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
        """
        Creates P Jigsaw puzzle from given image.
        :param img_label: the label that identifies the image in the collection
        :param permutations: set of permutations that shape each Jigsaw puzzle
        :return (imgs, labels): imgs is a tensor of P images where each image is a Jigsaw puzzle,
        labels is a tensor of P permutations that represent the label of the pretext task.
        """
        imgs = torch.ones((self.P, 3, 128, 128))
        labels = torch.ones(self.P, dtype=int)
        permutation_chosen_idx = [0 for _ in range(int(len(permutations)/2.5))]
        while len(permutation_chosen_idx) < self.P:
            random_choice = random.randint(0, len(permutations) - 1)
            permutation_chosen_idx.append(random_choice)
        random.shuffle(permutation_chosen_idx)
        for idx, p_idx in enumerate(permutation_chosen_idx):
            chosen_p = permutations[p_idx]
            dir_path = os.path.join(os.curdir, self.imgs_path)
            img_path = os.path.join(dir_path, img_label)
            img = Image.open(img_path)
            permuted_img = self.shuffle_tiles(img, chosen_p)
            imgs[idx] = permuted_img
            labels[idx] = p_idx
        return imgs, labels

    def shuffle_tiles(self, img, chosen_p):
        """
        This function is used to swap the tiles according to the chosen permutation.
        :param img: the image whose patches have to be swapped.
        :param chosen_p: the permutation to be used to swap the tiles of an image.
        :return img: the image that defines a specific Jigsaw puzzle according to a specific permutation.
        """
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
        """
        This function is used to calculate the coordinates of each tile to define
        9 patches from a given image.
        :param img: the image considered
        :param i: the i-th tile of the image.
        :return:
        """
        w = int(img.size[0] / self.N)
        y = int(i/self.N)
        x = i % self.N
        tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
        return tile

    def __getitem__(self, idx):
        """
        Returns a 4-tuple representing respectively: the jigsaw puzzles of a given image, the permutations associated
        with it, the image to be used for the segmentation, the ground truth mask.
        :param idx: identifies the idx-th image in the collection
        :return:
        """
        # each element in a batch is as follows [P_shuffled_imgs, idx_permutations, N_original_imgs, N_gt_imgs]
        seg_img, gt_img = self.get_segmentation_batch(idx)
        permutation_set = self.get_permutation_set()
        shuffled_imgs, perm_idx = self.build_data(self.imgs_labels[idx], permutation_set)
        return shuffled_imgs, perm_idx, seg_img, gt_img

    def __len__(self):
        # return 100
        return len(self.data)


class CustomDataPretext(Dataset):
    """
    This class defines the custom dataset used
    for the pretext task of the Personal Model.
    """


    def __init__(self, mode='train', split=[0.9, 0.1], full_data=False):
        super(CustomDataPretext, self).__init__()
        self.full_data = full_data
        if self.full_data:
            self.unlabelled_path = 'data/Resized_All/Unlabelled'
        else:
            self.unlabelled_path = 'data/Resized/Unlabelled'
        self.imgs_label = os.listdir(os.path.join(os.curdir, self.unlabelled_path))
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
        """
        Augment a specific image using a specific augmentation
        selected randomly from the set of augmentations proposed.
        :param idx: the index used to retrieve the idx-th image from the collection
        :return (image, augmentation_idx): the augmented image and the index of the augmentation applied (pseudo-label)
        """
        # image path
        image_label = self.data[idx]
        abs_path = os.path.join(os.curdir, self.unlabelled_path)
        img_path = os.path.join(abs_path, image_label)
        image = Image.open(os.path.join(os.curdir, img_path))
        augmentation_idx = random.randint(0, len(self.augmentations)-1)
        image = self.augmentations[augmentation_idx](image)
        return image, augmentation_idx

    def __getitem__(self, idx):
        """
        Returns the idx-th image from the collection that has been augmented, together with the index
        of the augmentation applied.
        :param idx: index used to access the idx-th image from the colleciton
        :return:
        """
        return self.process_batch(idx)

    def __len__(self):
        return len(self.data)
        # return 100

    def get_data(self):
        """
        Splits the data according to the split percentage.
        :return:
        """
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
