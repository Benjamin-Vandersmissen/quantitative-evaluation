import os
import numpy as np

fnames = [x.path for x in sorted(os.scandir('saliency/smoothgrad/'), key=lambda x: int(x.name.split('-')[0]))]
fnames = np.array(['-incorrect' not in fname for fname in fnames])
print(fnames.sum())
