import xml.etree.ElementTree as ET
import os
import json

import os

import PIL.Image
from PIL import ImageDraw
import torchvision
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
from tqdm import tqdm

from explain import explain
from saliency_methods import evaluate
import numpy as np
import utils


def visualize(img, bbox):
    img_draw = ImageDraw.ImageDraw(img)
    img_draw.rectangle(((bbox[1], bbox[3]), (bbox[2], bbox[4])), outline='red')
    img.show()


def mapping():
    labels = json.load(open("./imagenet_class_index.json","r"))
    mapping_ = {}
    for i in range(len(labels)):
        mapping_[labels[str(i)][0]] = i
    return mapping_


def rescale_bbox(bbox, img_size):
    xmin, xmax, ymin, ymax = bbox
    width, height = img_size

    #TODO : resize smaller edge to 256, other edge will keep the same ratio

    rescale_x = 0
    rescale_y = 0
    if width >= height:
        rescale_y = rescale_x = 256/height
    elif height > width:
        rescale_x = rescale_y = 256/width


    xmin, xmax = rescale_x * xmin, rescale_x * xmax
    ymin, ymax = rescale_y * ymin, rescale_y * ymax

    x_off = (rescale_x * width - 224)/2
    y_off = (rescale_y * height - 224)/2

    xmin, xmax = xmin - x_off, xmax - x_off
    ymin, ymax = ymin - y_off, ymax - y_off
    return xmin, xmax, ymin, ymax

def generate_bboxes(directory="val_bb"):
    class_mapping = mapping()
    fnames = [x.path for x in sorted(os.scandir(directory), key=lambda x: x.name)]

    bboxes = {}

    for i in range(len(fnames)):
        fname = fnames[i]
        xml = ET.parse(fname)
        fname = fname.split('/')[-1].split('.')[0]
        root = xml.getroot()
        objects = []
        img_width = int(root.find('size').find('width').text)
        img_height = int(root.find('size').find('height').text)

        for obj in root.findall('object'):
            name = class_mapping[obj.find('name').text]
            xmin = int(obj.find('bndbox').find('xmin').text)
            xmax = int(obj.find('bndbox').find('xmax').text)
            ymin = int(obj.find('bndbox').find('ymin').text)
            ymax = int(obj.find('bndbox').find('ymax').text)

            xmin, xmax, ymin, ymax = rescale_bbox((xmin, xmax, ymin, ymax), (img_width, img_height))
            objects.append((name, xmin, xmax, ymin, ymax))
        bboxes[fname] = objects
    return bboxes

bboxes = generate_bboxes()


def special_pointing_game(fname, saliency, class_idx, image=None):
    if len(saliency.shape) == 3:
        saliency = saliency.mean(axis=0)
    top_idx = np.argmax(saliency)
    top_idx = np.unravel_index(top_idx, saliency.shape)
    in_bbox = False
    to_pil = transforms.ToPILImage()
    for obj in bboxes[fname]:
        # visualize(to_pil(image.squeeze()), obj)
        if obj[0] == class_idx:
            if max(int(obj[3]),0) <= top_idx[0] <= min(int(obj[4])+1, 224) and max(int(obj[1]), 0) <= top_idx[1] <= min(int(obj[2])+1, 224):
                in_bbox = True

    return in_bbox


def make_baseline_saliency_map(fname, class_idx):
    saliency = np.zeros((1, 224, 224))
    for obj in bboxes[fname]:
        if obj[0] == class_idx:
            saliency[0, max(0,int(obj[3])):min(int(obj[4])+1, 224), max(0,int(obj[1])):min(int(obj[2])+1, 224)] = 1
    return saliency


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

transform2 = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])


validation = ImageFolder("val2", transform)


def calculate_pointing_scores():
    # methods=['baseline']
    methods = ['blurred_ig-zero', 'blurred_smoothgrad']
    # methods = ['gradcam', 'smoothgrad', 'occlusion', 'rise', 'ig-zero', 'lrp']
    for method in methods:
        testLoader = iter(DataLoader(validation))
        fnames = [x.path for x in sorted(os.scandir('saliency/vgg16/{}/'.format(method)), key=lambda x: int(x.name.split('-')[0]))]
        results = np.empty((len(fnames)))
        for filename, data in tqdm(enumerate(testLoader)):
            image, label = data
            label = label.squeeze().item()
            label = int(validation.classes[label])

            saliency = utils.normalize(np.load(fnames[filename])).astype(np.float32)
            fname = validation.imgs[filename][0].split('/')[-1].split('.')[0]
            results[filename] = special_pointing_game(fname, saliency, label, image)
        print(method, results.sum())
        np.save("pointing_{}".format(method), results)


def calculate_all_baselines():
    testLoader = iter(DataLoader(validation))
    for filename, data in tqdm(enumerate(testLoader)):
        image, label = data
        label = label.squeeze().item()
        label = int(validation.classes[label])

        fname = validation.imgs[filename][0].split('/')[-1].split('.')[0]
        baseline = make_baseline_saliency_map(fname, label)
        np.save("saliency/resnet50/baseline/{}-.npy".format(filename), baseline)


if __name__ == '__main__':
    calculate_pointing_scores()
    # calculate_all_baselines()
