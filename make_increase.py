import numpy as np
import os


def make_drops(prefix='evaluation/resnet50/drop_{}.npy', methods = None):
    if methods is None:
        methods = ['gradcam', 'ig-zero', 'lrp', 'occlusion', 'rise', 'smoothgrad']
    drops = []
    for name in methods:
        val = np.load(prefix.format(name))
        val = val.reshape((-1))
        drops.append(val)
    return np.array(drops)


if __name__ == '__main__':
    methods = ['gradcam', 'ig-zero', 'lrp', 'occlusion', 'rise', 'smoothgrad']
    # methods = ['blurred_ig-zero', 'blurred_smoothgrad', 'ig-zero', 'smoothgrad']
    for name in methods:
        drop = np.load('evaluation/test3/drop_{}_90.npy'.format(name.lower()))
        increase = np.load('evaluation/test3/increase_{}_90.npy'.format(name.lower()))

        fnames = [x.path for x in sorted(os.scandir('saliency/resnet50/{}/'.format(name)), key=lambda x: int(x.name.split('-')[0]))]
        fnames = np.array(['-incorrect' not in fname for fname in fnames])
        drop = drop[np.logical_not(np.isnan(drop))]
        drop = np.mean(drop)
        increase = np.mean(increase)
        print(name, drop, increase)
