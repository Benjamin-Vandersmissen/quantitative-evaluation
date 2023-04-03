import gc

from torch.utils.data import DataLoader

from utils import *

from saliency_methods.cam import GradCAM
from saliency_methods.decompose import LRP
from saliency_methods.occlusion import Occlusion
from saliency_methods.gradient import IntegratedGradients, Gradient
from saliency_methods.rise import Rise
from saliency_methods.composite import Smooth
from saliency_methods.mask import FullMask, MeanMask
from tqdm import tqdm

def explain(net, test_data, method: str = 'gradcam', device="cpu"):
    activation_maps = {}
    if method == 'rise':
        activation_maps['rise'] = helper(net, test_data, Rise(net, nr_masks=5000), device, "rise")
    if method == 'gradcam' or method == 'all':
        activation_maps['gradcam'] = helper(net, test_data, GradCAM(net), device, "gradcam")
    # if method == 'cam' or method == 'all':
    #     activation_maps['cam'] = cam_helper(net, test_data)
    if method == 'lrp' or method == 'all':
        activation_maps['lrp'] = helper(net, test_data, LRP(net), device, "lrp")
    if method == 'occlusion':
        activation_maps['occlusion'] = helper(net, test_data, Occlusion(net), device, "occlusion")
    if method == 'ig-zero':
        activation_maps['ig-zero'] = helper(net, test_data, IntegratedGradients(net, FullMask(0)), device, "ig-zero")
    if method == 'ig-mean':
        activation_maps['ig-mean'] = helper(net, test_data, IntegratedGradients(net, MeanMask(), device), "ig-mean")
    if method == 'smoothgrad':
        activation_maps['smoothgrad'] = helper(net, test_data, Smooth(Gradient(net)), device, "smoothgrad")

    for key, val in activation_maps.items():
        for key1, val1 in val.items():
            val1 = val1.sum(axis=0).squeeze()
            val[key1] = val1
        activation_maps[key] = val

    return activation_maps


def helper(net: nn.Module, test_data, saliency_method, device="cpu", method_name=""):
    testloader = iter(DataLoader(test_data, num_workers=0))
    acivation_maps = {}
    total_correct = 0

    print(saliency_method.__class__)
    mkdir_or_clear("saliency" + '/' + method_name)
    for i, data in tqdm(enumerate(testloader, 0)):
        inputs, labels = data
        input, label = inputs[0], labels[0]
        inputs, label = inputs.to(device), label.to(device)
        label = torch.LongTensor([int(test_data.classes[label.item()])])

        fname = str(i)
        top = np.argmax(net(inputs).cpu().detach().numpy())
        if top == label:
           fname += '-correct'
        else:
            fname += '-incorrect'

        activation_map = saliency_method.explain(inputs, label, only_positive=True, normalize=True, pixel_level=True)  # explain for the expected label
#            activation_map[activation_map <= 0] = 0
#            activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
        save_saliency_single(activation_map, fname, method_name)
        if i % 100 == 0:
            print(str(i) + "  -  " + str(label))

    return acivation_maps
