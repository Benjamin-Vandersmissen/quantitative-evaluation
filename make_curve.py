import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.metrics import auc
import os


def make_aucs(prefix='evaluation/vgg16/zero_region/deletion_{}.npy', methods = None):
    if methods is None:
        methods = ['gradcam', 'ig-zero', 'lrp', 'occlusion', 'rise', 'smoothgrad']
    aucs = []
    for name in methods:
        val = np.load(prefix.format(name))
        x = np.linspace(0, 1, val.shape[1])
        val = [auc(x, val[i]) for i in range(val.shape[0])]
        aucs.append(val)
    return np.array(aucs)


if __name__ == '__main__':

    beautiful_names = ['Grad-CAM', 'IG', 'LRP', 'Occlusion', 'RISE', 'Smoothgrad']
    names = ['gradcam', 'ig-zero', 'lrp', 'occlusion', 'rise', 'smoothgrad']
    blur_names = ['blurred_ig-zero', 'blurred_smoothgrad']
    blur_beautiful_names = ["IG (+blur)", 'Smoothgrad (+blur)']

    names += blur_names
    beautiful_names += blur_beautiful_names

    font = {'family': 'normal',
            'size': 12}

    matplotlib.rc('font', **font)
    cmap = plt.get_cmap("tab10")
    for i in range(len(names)):
        name = names[i]
        ins = np.load('evaluation/vgg16/coarsed_grads/delete_zero_pixel/deletion_{}.npy'.format(name.lower()))

        if 'new' in name:
            new_name = name.split('_')[0]
            print(ins[0])
        else:
            new_name = name

        mean = np.mean(ins, axis=0)
        x = np.linspace(0,100,ins.shape[1])
        label = "{} ({})".format(beautiful_names[i], np.round(auc(x, mean)/100,3))
        if 'blur' in name:
            style = '--'
        else:
            style = '-'
        if 'blur' in name:
            idx = names.index(name.split('blurred_')[1])
            color = cmap(idx)
        else:
            color = cmap(i)

        plt.plot(x, mean, color = color, linestyle=style, label=label)
    plt.legend()
    plt.xlim(0,100)
    plt.ylim(0,1)
    plt.ylabel("Prediction score")
    plt.xlabel("percentage deleted")
    plt.title("Deletion scores (VGG16)")
    #plt.tick_params(axis='x', which='both', top=False, bottom=False, labelbottom=False)
    plt.savefig("plot.pdf")
