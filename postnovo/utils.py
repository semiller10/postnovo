''' Functions used across project '''

import pickle as pkl
import json
import sys
import os

import postnovo.config as config

# import config

from urllib.request import urlopen
from shutil import copyfileobj


def save_pkl_objects(dir, **kwargs):
    for obj_name, obj in kwargs.items():
        verbose_print('saving', obj_name)
        with open(os.path.join(dir, obj_name + '.pkl'), 'wb') as f:
            pkl.dump(obj, f, 2)

def load_pkl_objects(dir, *args):
    return_list = []
    for obj_name in args:
        verbose_print('loading', obj_name)
        with open(os.path.join(dir, obj_name + '.pkl'), 'rb') as f:
            return_list.append(pkl.load(f))
    if len(args) == 1:
        return return_list[0]
    else:
        return tuple(return_list)

def save_json_objects(dir, **kwargs):
    for obj_name, obj in kwargs.items():
        verbose_print('saving', obj_name)
        with open(os.path.join(dir, obj_name + '.json'), 'w') as f:
            json.dump(obj, f)

def load_json_objects(dir, *args):
    return_list = []
    for obj_name in args:
        verbose_print('loading', obj_name)
        with open(os.path.join(dir, obj_name + '.json'), 'r') as f:
            return_list.append(json.load(f))
    if len(args) == 1:
        return return_list[0]
    else:
        return tuple(return_list)

def verbose_print(*args):
    if config.verbose[0]:
        for arg in args:
            print(arg, end = ' ')
        print()

def verbose_print_over_same_line(output_str):
    if config.verbose[0]:
        sys.stdout.write(output_str + '\r')
        sys.stdout.flush()