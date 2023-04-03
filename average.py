import numpy as np

avg = None
filename = "evaluation/test2/insertion_rise_uniform_region_{}_vgg.npy"
for i in range(5):
    run = np.load(filename.format(i))
    print(run)
    if avg is None:
        avg = run
    else:
        avg += run

avg = avg / 5
np.save(filename.format("mean"), avg)
