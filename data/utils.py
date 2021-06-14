import torch
import numpy as np
import torchvision
import os
from PIL import Image


def resize_images_unlabelled(folder_path, new_imgs_folder="./Resized/Unlabelled", img_size=512):
    # if the folder already exists, then return. Otherwise, create a new folder
    # with the resized images
    if os.path.exists(new_imgs_folder):
        if len(os.listdir(new_imgs_folder)) == 0:
            os.rmdir(new_imgs_folder)
        else:
            return
    os.mkdir(new_imgs_folder)
    for imgs_path in folder_path:
        tot_imgs = len(os.listdir(imgs_path))
        counter = 0
        for img_label in sorted(os.listdir(imgs_path)):
            if '.jpg' in img_label or '.png' in img_label:
                img_path = os.path.join(imgs_path, img_label)
                tensor_converter = torchvision.transforms.Compose([torchvision.transforms.Resize((512, 512)),
                                                               torchvision.transforms.ToTensor(),
                                                               torchvision.transforms.ToPILImage()])
                image = tensor_converter(Image.open(img_path))
                image.save(os.path.join(new_imgs_folder, img_label))
                if (counter + 1) % 500 == 0:
                    print("Processed [{}]/[{}]".format(counter, tot_imgs))
                counter += 1
        print('Folder Processed.')
    print('Done')




def resize_images_labelled(folder_path, new_imgs_folder="./Resized/Labelled", img_size=512):
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
                tensor_converter = torchvision.transforms.Compose([torchvision.transforms.Resize((512, 512)),
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


"""
The dataset comes with images that have gt for segmentation and others that don't. Namely, the images that
don't have gt are in the folders: 'ISIC2018_Task3_Training_Input', 'Testing/ISIC2018_Task1-2_Test_Input'.
Images that have gt labels are in the folders: 'Training/..', 'Validation/..'
The processing part aims at creating two folders: 'Labelled' and 'Unlabelled'. In the 'Labelled' folder, 
there will be two subfolders 'Images' and 'Ground_truth' which will contain the images and the respective
ground truths.
"""

path_unlabelled = ['/Users/tommasocapecchi/City/Master_Thesis/ISIC_2018/ISIC2018_Task3_Training_Input',
                   '/Users/tommasocapecchi/City/Master_Thesis/ISIC_2018/Testing/ISIC2018_Task1-2_Test_Input']

path_labelled = [('/Users/tommasocapecchi/City/Master_Thesis/ISIC_2018/Training/ISIC2018_Task1-2_Training_Input',
                  '/Users/tommasocapecchi/City/Master_Thesis/ISIC_2018/Training/ISIC2018_Task1_Training_GroundTruth'),
                 ('/Users/tommasocapecchi/City/Master_Thesis/ISIC_2018/Validation/ISIC2018_Task1-2_Validation_Input',
                  '/Users/tommasocapecchi/City/Master_Thesis/ISIC_2018/Validation/ISIC2018_Task1_Validation_GroundTruth')]

if os.path.exists('./Resized'):
    if len(os.listdir('./Resized')) == 0:
        os.rmdir('./Resized')
os.mkdir('./Resized')
resize_images_unlabelled(path_unlabelled)
resize_images_labelled(path_labelled)
