import os

import PIL.Image
import torchvision
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import torch
from tqdm import tqdm

from explain import explain
from saliency_methods import evaluate
from saliency_methods.mask import UniformMask
import numpy as np
import utils
import re

print(torch.__version__)
print(torchvision.__version__)

#torch.use_deterministic_algorithms(True)

loadsaliency = False

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])

validation = ImageFolder("val2", transform)

network = models.vgg16(pretrained=True)
network.eval()
for param in network.parameters():
    param.requires_grad_(False)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

network.to(device)

i = 0

loadsaliency = True

if loadsaliency:
    print("EVALUATING:")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    # methods = ['smoothgrad', 'lrp', 'occlusion']
    methods = ['blurred_ig-zero', 'blurred_smoothgrad']
    for eval_method in ['increase_drop']:
       with torch.no_grad():
            batch_size = 5
            for method in methods:
                print(eval_method, method)
                insertion = 0
                deletion = 0
                drop = 0
                increase = 0
                images = []
                labels = []
                testLoader = iter(DataLoader(validation))

                fnames = [x for x in os.scandir('saliency/vgg16/{}/'.format(method)) if re.match(".*\.npy\Z", x.path)]
                fnames = [x.path for x in sorted(fnames, key=lambda x: int(x.name.split('-')[0]))]
                for filename, data in enumerate(testLoader):
                    image, label = data
                    label = label.squeeze().item()
                    label = int(validation.classes[label])

                    images.append(image)
                    labels.append(label)

                    if len(images) == batch_size:
                        new_images = torch.concat(images, dim=0)
                        new_labels = torch.tensor(labels)
                        new_saliency = []
                        for i in range(filename-batch_size+1, filename+1):
                            load_correct = False
                            while not load_correct:
                                new_sall = None
                                try:
                                    print("loading {}".format(fnames[i]))
                                    new_sal = np.load(fnames[i])
                                    new_sal = new_sal.reshape((-1,224,224))
                                    load_correct = True
                                except:
                                    print("loading failed")
                                    pass

                            new_saliency.append(new_sal)
                            new_saliency[-1] = utils.normalize(new_saliency[-1])
                        new_saliency = np.stack(new_saliency).astype(np.float32)
                        if eval_method == "insertion":
                            new_insertion = evaluate.insertion(new_saliency, network, new_images, new_labels)
                            if not isinstance(insertion, np.ndarray):
                                insertion = new_insertion
                            else:
                                insertion = np.concatenate((insertion, new_insertion), 0)

                        elif eval_method == "increase_drop":
                            new_drop, new_increase = evaluate.average_drop_confidence_increase(new_saliency, network, new_images, new_labels)
                            new_increase = new_increase.detach().cpu().numpy()

                            if not isinstance(increase, np.ndarray):
                                drop = new_drop
                                increase = new_increase
                            else:
                                drop = np.concatenate((drop, new_drop), 0)
                                increase = np.concatenate((increase, new_increase), 0)

                        elif eval_method == 'deletion':
                            new_deletion = evaluate.deletion(new_saliency, network, new_images, new_labels, blur=UniformMask())
                            if not isinstance(deletion, np.ndarray):
                                deletion = new_deletion
                            else:
                                deletion = np.concatenate((deletion, new_deletion), 0)


                        images = []
                        labels = []
                if eval_method == "insertion":
                    np.save("insertion_{}.npy".format(method), insertion)
                elif eval_method == "increase_drop":
                    np.save("increase_{}.npy".format(method), increase)
                    np.save("drop_{}.npy".format(method), drop)
                elif eval_method == 'deletion':
                    np.save("deletion_{}.npy".format(method), deletion)
else:
    for method in ['lrp']:
        activation_maps = explain(network, validation, method, device)
        #utils.save_saliency(activation_maps)
    # for key, value in activation_maps.items():
    #     path = 'results/' + key + "/"
    #     utils.mkdir_or_clear(path)
    #
    #     testLoader = iter(DataLoader(validation))
    #     for i, data in enumerate(testLoader,0):
    #         img, label = data
    #         label = label.squeeze().item()
    #
    #         filename = i
    #         if filename in value:
    #             utils.heatmap(value[filename], filename=str(filename), directory=path)
