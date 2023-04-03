import torch
from PIL import Image
from matplotlib import cm
import numpy as np
import os
from torchvision import transforms
from tqdm import tqdm

def scantree(path):
    """Recursively yield DirEntry objects for given directory."""
    for entry in os.scandir(path):
        if entry.is_dir(follow_symlinks=False):
            yield from scantree(entry.path)  # see below for Python 2.x
        else:
            yield entry

def save_overlay(image, saliency_map, fname):
    """
    Overlay the saliency map on the image (with transparancy) and save it to a file.
    :param image: The image from which the saliency map is generated
    :param saliency_map: A saliency map
    :param fname: The filename without extension.
    :return: /
    """
    #saliency_map[saliency_map < saliency_map.mean()] = 0
    heatmap = np.uint8(255*cm.inferno(saliency_map))
    max = heatmap[:,:,:3].max(axis=2)
    mean = heatmap[:, :, :3].mean()
    heatmap[:, :, 3] = max >= mean
    heatmap[:, :, 3] *= 100
    heatmap[:, :, 3] += 50
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
        image = np.transpose(np.uint8(255*image), (1, 2, 0))

    if isinstance(image, np.ndarray):
        assert(heatmap.shape[:2] == image.shape[:2])

        image = Image.fromarray(image, mode='RGB')

    heatmap = Image.fromarray(heatmap).convert('RGBA')
    image.paste(heatmap, (0,0), heatmap)
    image.save(fname+".png")

trans = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

for name in ['blurred_ig-zero', 'blurred_smoothgrad']:
    fnames = [x.path for x in sorted(os.scandir('saliency/vgg16/{}/'.format(name.lower())), key=lambda x: int(x.name.split('-')[0]))]
    fnames2 = [x.path for x in sorted(scantree("val2"), key=lambda x : x.path)]

    for sal_name, img_name in tqdm(zip(fnames, fnames2)):
        img = trans(Image.open(img_name).convert('RGB'))
        sal = np.load(sal_name)
        sal[sal < 0] = 0

        # ONLY IF SMOOTHGRAD / IG:

        # percentile_99 = np.nanpercentile(sal, 99)
        # sal[sal >= percentile_99] = percentile_99

        if len(sal.shape) > 2:
            sal = sal.mean(0)
            sal = sal.reshape(*sal.shape[-2:])
        # sal = (sal-sal.min())/(sal.max()-sal.min())

        save_overlay(img, sal, "results/{}/{}".format(name, img_name.split('/')[-1].split('.')[0]))

