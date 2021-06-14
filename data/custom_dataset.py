from torch.utils.data.dataset import Dataset
import os
from PIL import Image
import torchvision
import enum
import matplotlib.pyplot as plt
import matplotlib.image as img_plot
import torch
import random

class Techniques(enum.IntEnum):
    CONTEXT_RESTORATION = 0
    CONTRASTIVE_LEARNING = 1
    JIGSAW = 2


class DatasetFactory:

    def build_dataset(self, technique, mode):
        if technique == Techniques.CONTEXT_RESTORATION:
            return ContextRestorationData(mode)
        elif technique == Techniques.CONTRASTIVE_LEARNING:
            # return ContrastiveLearningData(self.mode)
            pass
        else:
            # return JigsawData(self.mode)
            pass


class ContextRestorationData(Dataset):

    def __init__(self, mode='train'):
        self.curdir = '/Users/tommasocapecchi/City/Master_Thesis/ISIC_2018'
        self.unlabelled_data = self.build_pretext_data()
        self.labelled_data = self.build_task_data(mode)

    def build_pretext_data(self):
        """
        This function retrieve all the images' names and store it
        :return:
        """
        data = []
        path = os.path.join(self.curdir, 'ISIC2018_Task3_Training_Input')
        for idx, img_label in enumerate(os.listdir(path)):
            img_path = os.path.join(path, img_label)
            tensor_converter = torchvision.transforms.Compose([torchvision.transforms.Resize((512, 512)),
                                                               torchvision.transforms.ToTensor(),
                                                               torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                                                                (0.5, 0.5, 0.5))])
            original_img = tensor_converter(Image.open(img_path))
            data.append(original_img)
        return data

    def visualize_sample_image(self, n_images=4):
        """
        This function visualize n_images, sampled randomly
        from the dataset
        :param n_images: number of image to print per axis
        :return:
        """
        images = random.sample(self.unlabelled_data, n_images**2)
        plt.figure(figsize=(16, 16))
        for idx, img in enumerate(images):
            plt.subplot(n_images, n_images, idx + 1)
            image = image.permute(1, 2, 0)
            image = (image * 0.5) + 0.5
            plt.imshow(image)
            plt.axis('off')
        plt.show()
        plt.close()

    def build_task_data(self, mode):
        data = []
        if mode == 'train':
            path = os.path.join(self.curdir, 'Training')
            print(path)
        elif mode == 'val':
            path = os.path.join(self.curdir, 'Validation')
            print(path)
        elif mode == 'test':
            path = os.path.join(self.curdir, 'Testing')
            print(path)
        return data

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return len(self.data)


data_factory = DatasetFactory()
custom_dataset = data_factory.build_dataset(Techniques.CONTEXT_RESTORATION, 'train')
custom_dataset.visualize_sample_image()
