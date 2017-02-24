''' Useful functions across project '''

import pickle as pkl
import json
import sys

from config import test_dir

from os.path import realpath, dirname, join
from collections import OrderedDict

verbose = [True]

def save_pkl_objects(dir, **kwargs):
    for obj_name, obj in kwargs.items():
        verbose_print('saving', obj_name)
        with open(join(dir, obj_name + '.pkl'), 'wb') as f:
            pkl.dump(obj, f, 2)

def load_pkl_objects(dir, *args):
    return_list = []
    for obj_name in args:
        verbose_print('loading', obj_name)
        with open(join(dir, obj_name + '.pkl'), 'rb') as f:
            return_list.append(pkl.load(f))
    return tuple(return_list)

def save_json_objects(dir, **kwargs):
    for obj_name, obj in kwargs.items():
        verbose_print('saving', obj_name)
        with open(join(dir, obj_name + '.json'), 'w') as f:
            json.dump(obj, f)

def load_json_objects(dir, *args):
    return_list = []
    for obj_name in args:
        verbose_print('loading', obj_name)
        with open(join(dir, obj_name + '.json'), 'r') as f:
            return_list.append(json.load(f))
    return tuple(return_list)

def invert_dict_of_lists(d):
    values = set(a for b in d.values() for a in b)
    values = sorted(list(values))
    invert_d = OrderedDict((new_k, [k for k, v in d.items() if new_k in v]) for new_k in values)
    return invert_d

def verbose_print(*args):
    if verbose[0]:
        for arg in args:
            print(arg, end = ' ')
        print()

def verbose_print_over_same_line(output_str):
    if verbose[0]:
        sys.stdout.write(output_str + '\r')
        sys.stdout.flush()