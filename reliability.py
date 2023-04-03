import krippendorff
import numpy as np
from make_curve import make_aucs
from make_increase import make_drops
from scipy.stats import spearmanr, pointbiserialr, pearsonr


def ranked_interrater_relevance(values):
    # Assume input matrix of shape (nr_observations, nr_raters).
    # In this case, each image is a rater, for each rater we determine the ranking of the explanation methods folowing the metric.
    # This means that if we have three methods and the performance has the following ordering : method 2 < method 1 < method 3
    # Then the observations for this rater are [1,2,0] i.e. the first observation (method 1) is better than the second (method 2) but worse than the last (method 3)

    ranked_values = np.argsort(values, axis=0).swapaxes(0, 1)
    return krippendorff.alpha(ranked_values)


aucs_ins = make_aucs('evaluation/vgg16/insert_uniform_region/insertion_{}.npy')
aucs_ins2 = make_aucs('evaluation/vgg16/insert_blur_region/insertion_{}.npy')
aucs_del = make_aucs('evaluation/vgg16/delete_uniform_region/deletion_{}.npy')
aucs_del2 = make_aucs('evaluation/vgg16/delete_blur_region/deletion_{}.npy')
drops = make_drops('evaluation/resnet50/drop_binary/90/drop_{}.npy')
drops2 = make_drops('evaluation/resnet50/drop_{}.npy')
pointing = make_drops("evaluation/resnet50/pointing_{}.npy")
# pointing2 = make_drops("evaluation/resnet50/pointing_{}.npy")

drops_nan = np.isnan(drops)
drops[drops_nan] = 0

# TODO: VERY IMPORTANT !!!! With Drop or deletion, the lower values are better, while for insertion higher values are better
#  We can easily fix this by transforming Drop & deletion to a value where the higher is better by doing 1 - drop / 1 - deletion.

methods = ['gradcam', 'ig-zero', 'lrp', 'occlusion', 'rise', 'smoothgrad']
# methods = ['ig-zero', 'smoothgrad']
for i in range(len(methods)):
    print(methods[i], spearmanr(drops[i], drops[i], nan_policy='omit').correlation)

    # TODO: IF USING pointing, use pointbiserialr
    # print(methods[i], pointbiserialr(pointing[i], 1-drops[i]).correlation)
    # TODO: IF USING DOUBLE BINARY, use pearsonr
    # print(methods[i], pearsonr(pointing[i], pointing2[i]))

# print(ranked_interrater_relevance(aucs))
