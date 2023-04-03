from os import listdir, mkdir
import os
import shutil
import json
import random

files = listdir("validation")
files.sort()
labels = [label.strip() for label in open("mapping.txt", "r").readlines()]
files = list(zip(labels, files))

sampled_labels = list(set(labels))
random.shuffle(sampled_labels)
sampled_labels = set(sampled_labels[:100])

files_dict = dict()
for f in files:
    if f[0] not in sampled_labels:
        continue
    if f[0] in files_dict:
        files_dict[f[0]].append(f)
    else:
        files_dict[f[0]] = [f]

files = [file for f in files_dict.items() for file in f[1]]

new_files = []
# for f in files_dict.items():
#     bla = f[1]
#     random.shuffle(bla)
#     new_files += bla[:5]
# files = new_files

#random.shuffle(files)
#files = files[:5000]

def create_dir_if_possible(path):
    if (not os.path.isdir(path)):
        mkdir(path)

create_dir_if_possible("val2")

temp = json.load(open("imagenet_class_index.json","r"))
mapping = dict()
arranged = [i for i in range(1000)]
arranged.sort()
for i in range(1000):
    if temp[str(i)][0] not in sampled_labels:
        continue
    mapping[temp[str(i)][0]] = str(i)
    create_dir_if_possible(os.path.join("val2", str(i)))

for pair in files:
    shutil.copyfile(os.path.join("validation", pair[1]), os.path.join("val2", mapping[pair[0]], pair[1]))
