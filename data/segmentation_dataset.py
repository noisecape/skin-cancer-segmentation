from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
import numpy as np


class SegmentationDataset(Dataset):

    imgs_path = 'data/Resized/Labelled/Images'
    gt_path = 'data/Resized/Labelled/Groundtruth'

    def __init__(self, mode, split_perc=[0.7, 0.1, 0.2]):
        super(SegmentationDataset, self).__init__()
        self.mode = mode
        self.imgs_labels = sorted(os.listdir(os.path.join(os.curdir, SegmentationDataset.imgs_path)))
        self.gt_labels = sorted(os.listdir(os.path.join(os.curdir, SegmentationDataset.gt_path)))
        if mode == 'train':
            start_idx = 0
            end_idx = int(len(self.imgs_labels) * split_perc[0])
            self.data = self.build_data(start_idx, end_idx)
        elif mode == 'val':
            start_idx = int(len(self.imgs_labels) * split_perc[0])
            end_idx = start_idx + int(len(self.imgs_labels) * split_perc[1])
            self.data = self.build_data(start_idx, end_idx)
        elif mode == 'test':
            start_idx = int(len(self.imgs_labels) * (split_perc[0] + split_perc[1]))
            end_idx = len(self.imgs_labels)
            self.data = self.build_data(start_idx, end_idx)

    def build_data(self, start_idx, end_idx):
        data = [(img, gt) for img, gt in zip(self.imgs_labels[start_idx:end_idx], self.gt_labels[start_idx:end_idx])]
        return data

    def __getitem__(self, idx):
        img_path = os.path.join(SegmentationDataset.imgs_path, self.data[idx][0])
        gt_path = os.path.join(SegmentationDataset.gt_path, self.data[idx][1])
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

    def __len__(self):
        # return len(self.data)
        return 100

    def visualize_image(self, x):
        plt.figure(figsize=(16, 16))
        x = x.permute(1, 2, 0)
        x = (x * 0.5) + 0.5
        plt.imshow(x)
        plt.axis('off')
        plt.show()
        plt.close()


# dataset = SegmentationDataset(mode='train')
# dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
# for batch in dataloader:
#     print(batch[0])
#     print(batch[1])
