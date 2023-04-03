from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import os

import skimage.io as img_io

#Normalise and then threshold

def threshold_all(heatmaps, percentile=80, value=-1, global_percentile=False):
    retvalue = {}
    i = 0

    if global_percentile:
        for key, values in heatmaps.items():
            global_threshold = np.percentile(np.concatenate(list(values.values())), percentile)
            temp = {}
            for key2, heatmap in values.items():
                temp[key2] = threshold(heatmap, value=global_threshold)
            retvalue[key] = temp
    else:
        for key, values in heatmaps.items():
            temp = {}
            for key2, heatmap in values.items():
                if value != -1:  # normalize for value based thresholding.
                    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                temp[key2] = threshold(heatmap, percentile, value)
            retvalue[key] = temp
    return retvalue


# Normalise and then threshold

def threshold(heatmap, percentile=80, value=-1):
    base = value
    if value == -1:
        base = np.percentile(heatmap, percentile)
    heatmap[heatmap < base] = 0
    return heatmap


def tensor_to_image_format(tensor):
    out = tensor.clone().squeeze(0)
    out = np.rollaxis(out.detach().numpy(), 0, 3)
    return out


def resize_threshold_percentile(heatmap, shape, smooth=True, percentile=None):
    if not smooth:
        heatmap = resize(heatmap, shape, order=0)
    else:
        heatmap = resize(heatmap, shape)
    if not(percentile is None):
        percentile = np.percentile(heatmap, percentile)
    else:
        percentile = 0.0

    heatmap[heatmap <= percentile] = 0
    return heatmap


def overlay_heatmap(img, heatmap):
    # img = img.mean(axis=2)  # Necessary for grayscale
    #plt.imshow(img)

    heatmap = heatmap.copy()
    #heatmap[np.isclose(heatmap, 0, atol=5e-5)] = np.nan  # set all 0-values to NaN, use an epsilon
    plt.imshow(heatmap, alpha=1, cmap='jet')  # don't display the NaN values


def show_heatmap(img, heatmap):
    overlay_heatmap(img, heatmap)
    plt.show()


def save_heatmap_overlay(img, heatmap, name, directory='results/'):
    overlay_heatmap(img, heatmap)
    plt.axis('off')
    plt.savefig(fname=directory + name + ".png", bbox_inches='tight')
    plt.clf()


def save_heatmap(heatmap, name, directory='results/'):
    plt.imsave(directory + name + ".png", heatmap, format='png')


def mkdir_or_clear(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        for root, dirs, files in os.walk(path):
            for file in files:
                os.remove(os.path.join(root, file))


# Convert the filename of a explanation map to the related mask
def filename_to_mask(filename, mask_dir):
    split = filename.split('-')
    mask_filename = mask_dir + '/' + split[0] + '/' + split[1] + ".png"
    return file_to_np_array(mask_filename)[0,:,:]  # all three channels are equal, so take only 1


def file_to_np_array(filename):
    image = img_io.imread(filename)
    image = resize(image, (128, 128))
    image = np.rollaxis(image, 2, 0)  # move the channel dimension to the first position
    image = image.astype('float32')  # Convert to float
    return image


def get_layers(net):  # replace linear layers with convolutional layers
    layers = []
    func = lambda module, input, output: layers.append(module)

    handles = []

    for name, m in net.named_modules():
        if name != '':  # Otherwise we have the last activation double
            handles.append(m.register_forward_hook(func))

    net.forward(torch.zeros(1, 3, 128, 128))
    for handle in handles:
        handle.remove()

    new_layers = []
    first_linear = True
    for i in range(len(layers)):
        layer = layers[i]
        if isinstance(layer, nn.Linear):
            new_layer = None
            if first_linear:  # We forego the flatten layer, so this conv layer needs to flatten as well
                m, n = 256, layer.weight.shape[0]
                new_layer = nn.Conv2d(m, n, 6)  # After the final convolution, we have 6x6
                new_layer.weight = nn.Parameter(layer.weight.reshape(n, m, 6, 6))
                first_linear = False
            else:
                m,n = layer.weight.shape[1], layer.weight.shape[0]
                new_layer = nn.Conv2d(m, n, 1)
                new_layer.weight = nn.Parameter(layer.weight.reshape(n, m, 1, 1))
            new_layer.bias = nn.Parameter(layer.bias)
            layers[i] = new_layer

        if not isinstance(layers[i], nn.Flatten):
            new_layers.append(layers[i])
    return new_layers


def newlayer(layer, func):  # Use the function func on all parameters
    import copy
    layer = copy.deepcopy(layer)

    try:
        layer.weight = nn.Parameter(func(layer.weight))
    except AttributeError:
        pass

    try:
        layer.bias = nn.Parameter(func(layer.bias))
    except AttributeError:
        pass

    return layer


# returns the indices of the heatmap sorted by maximum value first
def importance(heatmap):
    index = np.array(np.unravel_index(np.argsort(- heatmap, axis=None), heatmap.shape))
    return index


def save_saliency_single(saliency, name, typ, parent_directory="./saliency"):
    with open(parent_directory +"/" + typ + "/" + name + ".npy", "wb") as f:
        np.save(f, saliency)

def save_saliency(saliency, parent_directory='./saliency/'):
    for typ, maps in saliency.items():
        mkdir_or_clear(parent_directory + '/' + typ)

        for name, activation_map in maps.items():

            #new_name = name.split('/')[-2] + ' - ' + name.split('/')[-1].split('.png')[0]
            np.save(parent_directory + '/' + typ + '/' + str(name), activation_map)


def load_explanation(fname, parent_directory="./saliency/", method="GradCam"):
    path = os.path.join(parent_directory, method.lower(), fname)
    return np.load(path)

def load_saliency(image_directory, parent_directory="./saliency/"):
    saliency = {}
    for entry in os.scandir(parent_directory + '/'):
        if entry.is_dir():
            saliency[entry.name] = {}

            for entry2 in os.scandir(parent_directory  + '/' + entry.name +'/'):
                filename = entry2.name.split('.npy')[0].split(' - ')[-1]
                new_name = int(filename)
                saliency[entry.name][new_name] = np.load(entry2.path)
    return saliency


def heatmap(R, directory, filename):

    b = 10*((np.abs(R)**3.0).mean()**(1.0/3))

    from matplotlib.colors import ListedColormap
    my_cmap = plt.cm.seismic(np.arange(plt.cm.seismic.N))
    my_cmap[:,0:3] *= 0.85
    my_cmap = ListedColormap(my_cmap)
    plt.axis('off')
    plt.imshow(R,cmap=my_cmap,vmin=-b,vmax=b,interpolation='nearest')
    #plt.show()
    plt.savefig(fname=directory + filename + ".png", bbox_inches='tight')

def normalize(sal):
    sal[sal < 0] = 0
    sal = (sal-sal.min())/np.maximum((sal.max()-sal.min()), 5e-6)
    return sal
