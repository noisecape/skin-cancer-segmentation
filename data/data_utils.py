import torch
import numpy as np
import torchvision
import os
from PIL import Image
import csv

IMAGE_SIZE = 128


def resize_images_unlabelled(folder_path, cancer_imgs, new_imgs_folder="./Resized/Unlabelled", img_size=IMAGE_SIZE):
    # if the folder already exists, then return. Otherwise, create a new folder
    # with the resized images
    if os.path.exists(new_imgs_folder):
        if len(os.listdir(new_imgs_folder)) == 0:
            os.rmdir(new_imgs_folder)
        else:
            return
    os.mkdir(new_imgs_folder)
    for idx, imgs_path in enumerate(folder_path):
        # from the HAM10000 filters only the images relative to skin cancer lesions.
        if idx == 0:
            tot_imgs = len(os.listdir(imgs_path))
            counter = 0
            for cancer in cancer_imgs:
                cancer = cancer+'.jpg'
                img_path = os.path.join(imgs_path, cancer)
                tensor_converter = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size)),
                                                                   torchvision.transforms.ToTensor(),
                                                                   torchvision.transforms.ToPILImage()])
                image = tensor_converter(Image.open(img_path))
                image.save(os.path.join(new_imgs_folder, cancer))
                if (counter + 1) % 500 == 0:
                    print("Processed [{}]/[{}]".format(counter, tot_imgs))
                counter += 1
            print('Folder Processed.')
        else:
            tot_imgs = len(os.listdir(imgs_path))
            counter = 0
            for img_label in sorted(os.listdir(imgs_path)):
                if '.jpg' in img_label or '.png' in img_label:
                    img_path = os.path.join(imgs_path, img_label)
                    tensor_converter = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size)),
                                                                       torchvision.transforms.ToTensor(),
                                                                       torchvision.transforms.ToPILImage()])
                    image = tensor_converter(Image.open(img_path))
                    image.save(os.path.join(new_imgs_folder, img_label))
                    if (counter + 1) % 500 == 0:
                        print("Processed [{}]/[{}]".format(counter, tot_imgs))
                    counter += 1
            print('Folder Processed.')
    print('Done')


def resize_images_labelled(folder_path, new_imgs_folder="./Resized/Labelled", img_size=IMAGE_SIZE):
    if not os.path.exists(new_imgs_folder):
        os.mkdir(new_imgs_folder)
    if not os.path.exists(os.path.join(new_imgs_folder, 'Images')):
        os.mkdir(os.path.join(new_imgs_folder, 'Images'))
    if not os.path.exists(os.path.join(new_imgs_folder, 'Groundtruth')):
        os.mkdir(os.path.join(new_imgs_folder, 'Groundtruth'))

    for imgs_path in folder_path:
        tot_imgs = len(os.listdir(imgs_path[0]))
        counter = 0
        for img, gt in zip(sorted(os.listdir(imgs_path[0])), sorted(os.listdir(imgs_path[1]))):
            if '.jpg' in img or '.png' in img and '.jpg' in gt or '.png' in gt:
                tensor_converter = torchvision.transforms.Compose([torchvision.transforms.Resize((img_size, img_size)),
                                                                   torchvision.transforms.ToTensor(),
                                                                   torchvision.transforms.ToPILImage()])
                image = tensor_converter(Image.open(os.path.join(imgs_path[0], img)))
                ground_truth = tensor_converter(Image.open(os.path.join(imgs_path[1], gt)))

                new_img_path = os.path.join(new_imgs_folder, 'Images')
                new_gt_path = os.path.join(new_imgs_folder, 'Groundtruth')

                image.save(os.path.join(new_img_path, img))
                ground_truth.save(os.path.join(new_gt_path, gt))
                if (counter + 1) % 500 == 0:
                    print("Processed [{}]/[{}]".format(counter, tot_imgs))
                counter += 1
        print('Folder Processed.')
    print('Done')


def plot_data_dimensions(path_unlabelled, path_labelled):
    imgs_sizes = []
    # process unlabelled images
    for imgs_batch in path_unlabelled:
        for img_label in os.listdir(imgs_batch):
            if '.jpg' in img_label or '.png' in img_label:
                full_label = os.path.join(imgs_batch, img_label)
                img = Image.open(full_label)
                imgs_sizes.append((img.size[0], img.size[1]))
    # process labelled images
    for imgs_batch in path_labelled:
        for imgs_label in sorted(os.listdir(imgs_batch[0])):
            if '.jpg' in imgs_label or '.png' in imgs_label and '.jpg':
                img_label = os.path.join(imgs_batch[0], imgs_label)
                img = Image.open(img_label)
                imgs_sizes.append((img.size[0], img.size[1]))

    # compute statistics
    avg_width = np.mean([img_size[0] for img_size in imgs_sizes])
    avg_height = np.mean([img_size[1] for img_size in imgs_sizes])

    median_width = np.median([img_size[0] for img_size in imgs_sizes])
    median_height = np.median([img_size[1] for img_size in imgs_sizes])
    # plot histogram of both median and average values for width and height

"""
The dataset comes with images that have gt for segmentation and others that don't. Namely, the images that
don't have gt are in the folders: 'ISIC2018_Task3_Training_Input', 'Testing/ISIC2018_Task1-2_Test_Input'.
Images that have gt labels are in the folders: 'Training/..', 'Validation/..'
The processing part aims at creating two folders: 'Labelled' and 'Unlabelled'. In the 'Labelled' folder, 
there will be two subfolders 'Images' and 'Ground_truth' which will contain the images and the respective
ground truths.
"""

# PATH FOR MAC
# path_unlabelled = ['/Users/tommasocapecchi/City/Master_Thesis/ISIC_2018/ISIC2018_Task3_Training_Input',
#                    '/Users/tommasocapecchi/City/Master_Thesis/ISIC_2018/Testing/ISIC2018_Task1-2_Test_Input']

# path_labelled = [('/Users/tommasocapecchi/City/Master_Thesis/ISIC_2018/Training/ISIC2018_Task1-2_Training_Input',
#                   '/Users/tommasocapecchi/City/Master_Thesis/ISIC_2018/Training/ISIC2018_Task1_Training_GroundTruth'),
#                  ('/Users/tommasocapecchi/City/Master_Thesis/ISIC_2018/Validation/ISIC2018_Task1-2_Validation_Input',
#                   '/Users/tommasocapecchi/City/Master_Thesis/ISIC_2018/Validation/ISIC2018_Task1_Validation_GroundTruth')]

# PATH FOR WINDOWS
path_unlabelled = ['C:/Users/Noisecape/PycharmProjects/ISIC_2018/ISIC2018_Task3_Training_Input',
                   'C:/Users/Noisecape/PycharmProjects/ISIC_2018/Testing/ISIC2018_Task1-2_Test_Input']

path_labelled = [('C:/Users/Noisecape/PycharmProjects/ISIC_2018/Training/ISIC2018_Task1-2_Training_Input',
                 'C:/Users/Noisecape/PycharmProjects/ISIC_2018/Training/ISIC2018_Task1_Training_GroundTruth'),
                 ('C:/Users/Noisecape/PycharmProjects/ISIC_2018/Validation/ISIC2018_Task1-2_Validation_Input',
                  'C:/Users/Noisecape/PycharmProjects/ISIC_2018/Validation/ISIC2018_Task1_Validation_GroundTruth')]

path_gt_csv = 'C:/Users/Noisecape/PycharmProjects/ISIC_2018/csv_ham10000.csv'
skin_cancer_imgs = []
with open(path_gt_csv) as csv_file:
    csv = csv.reader(csv_file)
    for row in csv:
        if row[1] == '1' or row[3] == '1':
            skin_cancer_imgs.append(row[0])


# plot_data_dimensions(path_unlabelled, path_labelled)

if os.path.exists('./Resized'):
    if len(os.listdir('./Resized')) == 0:
        os.rmdir('./Resized')
else:
    os.mkdir('./Resized')
    resize_images_unlabelled(path_unlabelled, skin_cancer_imgs)
    resize_images_labelled(path_labelled)
