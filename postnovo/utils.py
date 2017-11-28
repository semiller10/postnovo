''' Functions used across project '''

import json
import os
import pandas as pd
import pickle as pkl
import sys

from multiprocessing import current_process
from shutil import copyfileobj
from urllib.request import urlopen

if 'postnovo' in sys.modules:
    import postnovo.config as config
else:
    import config

progress_count = 0

def save_dfs(dir, **kwargs):
    for df_name, df in kwargs.items():
        verbose_print('saving', df_name)
        df.to_csv(os.path.join(dir, df_name + '.tsv'), sep='\t', index=False)

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
        # Clear to end of line
        sys.stdout.write('\033[K')
        sys.stdout.write(output_str + '\r')
        sys.stdout.flush()

def remove_mod_chars(seq):

    mod_chars = config.mod_chars

    del_list = []
    in_parens = False
    for char in seq:
        if char == '(' or char == '[':
            in_parens = True

        if in_parens:
            del_list.append(True)
        elif char in mod_chars:
            del_list.append(True)
        else:
            del_list.append(False)

        if char == ')' or char == ']':
            in_parens = False

    cleaned_seq = ''
    for i, is_deleted in enumerate(del_list):
        if is_deleted == False:
            cleaned_seq += seq[i]

    return cleaned_seq

def print_percent_progress_multithreaded(procedure_str, one_percent_total_count, cores):

    if current_process()._identity[0] % cores == 1:
        global progress_count
        progress_count += 1
        if int(progress_count % one_percent_total_count) == 0:
            percent_complete = int(progress_count / one_percent_total_count)
            if percent_complete <= 100:
                verbose_print_over_same_line(procedure_str + str(percent_complete) + '%')

def print_percent_progress_singlethreaded(procedure_str, one_percent_total_count):

    global progress_count
    progress_count += 1
    if int(progress_count % one_percent_total_count) == 0:
        percent_complete = int(progress_count / one_percent_total_count)
        if percent_complete <= 100:
            verbose_print_over_same_line(procedure_str + str(percent_complete) + '%')

# Currently unused:
def invert_dict_of_lists(d):
    values = set(a for b in d.values() for a in b)
    values = sorted(list(values))
    invert_d = OrderedDict((new_k, [k for k, v in d.items() if new_k in v]) for new_k in values)
    return invert_d