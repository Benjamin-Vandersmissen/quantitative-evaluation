import numpy as np
import os
from tqdm import tqdm

for net in ['vgg16']:
    for name in ['rise']:
        fnames = [x.path for x in sorted(os.scandir('saliency/{}/{}/'.format(net, name.lower())), key=lambda x: int(x.name.split('-')[0]))]

        for sal_name in tqdm(fnames):
            sal = np.load(sal_name)
            sal = sal.mean(axis=0, keepdims=True)
            sal[sal < 0] = 0
            sal = (sal - sal.min())/(sal.max()-sal.min())
            np.save(sal_name, sal)
