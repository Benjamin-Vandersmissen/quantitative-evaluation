import numpy as np
import torch
from torchvision.transforms import GaussianBlur
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from sklearn.neighbors import KernelDensity


def count_median_attribution():
    for net in['vgg16', 'resnet50']:
        for name in ['ig-zero', 'smoothgrad', 'occlusion', 'lrp', 'gradcam', 'rise']:
            fnames = [x.path for x in sorted(os.scandir('saliency/{}/{}/'.format(net, name.lower())), key=lambda x: int(x.name.split('-')[0]))]
            all_sal = []
            for sal_name in tqdm(fnames):
                sal = np.load(sal_name)
                all_sal.append(sal)
            all_sal = np.asarray(all_sal)
            print("median {}, {}  =  {}".format(net, name, np.nanmedian(all_sal)))
            print("mean {}, {}  =  {}".format(net, name, np.nanmean(all_sal)))
            print("25quant {}, {}  =  {}".format(net, name, np.nanquantile(all_sal, q=.25)))
            print("75quant {}, {}  =  {}".format(net, name, np.nanquantile(all_sal, q=.75)))

def plot_stats(load=False):
    font = {'family': 'normal',
            'size': 14}

    matplotlib.rc('font', **font)
    for net in['resnet50']:
        beautiful_names = ['Grad-CAM', 'IG', 'LRP', 'Occlusion', 'RISE', 'Smoothgrad']
        names = ['gradcam', 'ig-zero', 'lrp', 'occlusion', 'rise', 'smoothgrad']
        x_axis = np.linspace(0, 1, 100)
        for i in range(len(names)):
            name = names[i]
            fnames = [x.path for x in sorted(os.scandir('saliency/{}/{}/'.format(net, name.lower())), key=lambda x: int(x.name.split('-')[0]))]
            all_sal = np.empty((5000*224*224))
            if not load:
                for j, sal_name in tqdm(enumerate(fnames)):
                    if j == 5000:
                        break
                    sal = np.load(sal_name)
                    sal = sal.reshape(-1)
                    all_sal[sal.shape[0]*j:sal.shape[0]*(j+1)] = sal
                all_sal[np.isnan(all_sal)] = 0
                kde = KernelDensity(bandwidth=0.05, kernel='gaussian', rtol=0.01)
                kde.fit(all_sal[:, None])
                logprob = kde.score_samples(x_axis[:, None])
                prob = np.exp(logprob)
                np.save(name + ".npy", np.exp(logprob))
            else:
                prob = np.load(name + ".npy")
            plt.plot(x_axis, prob, label=beautiful_names[i])
        plt.xlabel("relevance")
        plt.ylabel("density")
        plt.xlim((0,1))
        plt.ylim((0,7.5))
        plt.legend()
        plt.savefig("density_resnet"+".pdf")

def blur_saliency():

    blur = GaussianBlur(11, 5)

    for net in ['vgg16']:
        for name in ['ig-zero', 'smoothgrad']:
            fnames = [x.path for x in sorted(os.scandir('saliency/{}/blurred_{}/'.format(net, name.lower())), key=lambda x: int(x.name.split('-')[0]))]

            for sal_name in tqdm(fnames):
                sal = np.load(sal_name)
                sal[sal < 0] = 0
                sal = (sal - sal.min())/(sal.max()-sal.min())
                blurred = blur(torch.from_numpy(sal)).detach().cpu().numpy()
                blurred = (blurred - blurred.min())/(blurred.max()-blurred.min())
                np.save(sal_name, blurred)


plot_stats(True)
