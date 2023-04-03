import numpy as np
import os
from tqdm import tqdm

for net in ['vgg16', 'resnet50']:
    for name in ['ig-zero', 'smoothgrad']:
        fnames = [x.path for x in sorted(os.scandir('saliency/{}/{}/'.format(net, name.lower())), key=lambda x: int(x.name.split('-')[0]))]

        for sal_name in tqdm(fnames):
            sal = np.load(sal_name)
            sal[sal < 0] = 0
            percentile_99 = np.nanpercentile(sal, 99)
            sal[sal >= percentile_99] = percentile_99
            sal = (sal - sal.min())/(sal.max()-sal.min())
            np.save(sal_name, sal)
