from config import *

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import sys
import time
import datetime
import warnings
warnings.filterwarnings("ignore")

from utils import (save_pkl_objects, load_pkl_objects,
                   save_json_objects, load_json_objects)

from functools import partial
from itertools import product
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from os.path import join
from multiprocessing import Pool
from collections import Counter
from scipy.stats import norm

def classify(prediction_df, train, ref_file, cores, alg_list):

    if train:
        #prediction_df = find_target_accuracy(prediction_df, ref_file, cores)

        #prediction_df = standardize_prediction_df_cols(prediction_df)
        #save_pkl_objects(test_dir, **{'prediction_df_test': prediction_df})
        prediction_df, = load_pkl_objects(test_dir, 'prediction_df_test')

        subsampled_df = subsample_training_data(prediction_df, alg_list, cores)

        #training_df = update_training_data(prediction_df)
        training_df, = load_pkl_objects(training_dir, 'training_df_test')

        make_training_forests(training_df, alg_list, cores)
        save_pkl_objects(training_dir, 'training_model_test')

    else:
        pass

def subsample_training_data(prediction_df_orig, alg_list, cores):

    ### TEMPORARY ###
    cores = 3
    #################

    subsample_size = 1000
    subsample_row_indices = []

    subsampled_alg_group_df_dict = {}
    multiindex_groups = list(product((0, 1), repeat = len(alg_list)))[1:]

    prediction_df = prediction_df_orig.copy()
    prediction_df.drop(accepted_mass_tols + ['is top rank single alg', 'seq'],
                       axis = 1, inplace = True)

    accuracy_division = 10
    accuracy_bins = sorted([round(x / accuracy_division, 1) for x in range(accuracy_division)], reverse = True)
    
    lower = 0
    upper = 3
    weight_bins = np.arange(lower, upper + (upper - lower) / accuracy_division, (upper - lower) / accuracy_division)
    sigma = 0.9
    mu_location = 0.5
    accuracy_weights = (norm.cdf(weight_bins[1: 1 + accuracy_division], loc = mu_location, scale = sigma)
                        - norm.cdf(weight_bins[: accuracy_division], loc = mu_location, scale = sigma))\
                            / (norm.cdf(upper, loc = mu_location, scale = sigma)
                               - norm.cdf(lower, loc = mu_location, scale = sigma))
    accuracy_subsample_weights = {acc_bin: weight for acc_bin, weight in zip(accuracy_bins, accuracy_weights)}
    accuracy_subsample_sizes = {acc_bin: int(weight * subsample_size) for acc_bin, weight in accuracy_subsample_weights.items()}
    while sum(accuracy_subsample_sizes.values()) != subsample_size:
        accuracy_subsample_sizes[accuracy_bins[0]] += 1

    for multiindex in multiindex_groups:
        multiindex_list = list(multiindex)
        alg_group_df_key = tuple([alg for i, alg in enumerate(alg_list) if multiindex[i]])
        alg_group_df = prediction_df.xs(multiindex).reset_index().set_index(['scan'])
        alg_group_df.dropna(1, inplace = True)

        if alg_group_df.shape[0] > subsample_size:
            pipe = make_pipeline(StandardScaler(),
                                 PCA(n_components = int(alg_group_df.columns.size / 4)),
                                 KMeans(n_clusters = int(alg_group_df.shape[0] / 20), n_jobs = cores))

            cluster_assignments = pipe.fit_predict(alg_group_df.as_matrix())
            cluster_assignment_accuracies = zip(cluster_assignments, alg_group_df['ref match'])
            sum_cluster_accuracies = {}.fromkeys(cluster_assignments, 0)
            for cluster, acc in cluster_assignment_accuracies:
                sum_cluster_accuracies[cluster] += acc
            cluster_counts = Counter(cluster_assignments)
            mean_cluster_accuracies = {}.fromkeys(sum_cluster_accuracies, 0)
            for cluster in cluster_counts:
                mean_cluster_accuracies[cluster] = min(
                    accuracy_bins,
                    key = lambda accuracy_bin: abs(accuracy_bin - int(
                        sum_cluster_accuracies[cluster] * 10 / cluster_counts[cluster]) / 10))
            ordered_clusters_accuracies = sorted(
                mean_cluster_accuracies.items(), key = lambda cluster_accuracy_tuple: cluster_accuracy_tuple[1], reverse = True)
            cluster_assignments_row_indices = [(cluster, index) for index, cluster in enumerate(cluster_assignments)]
            cluster_row_indices_dict = {cluster: [] for cluster in mean_cluster_accuracies}
            for cluster, index in cluster_assignments_row_indices:
                cluster_row_indices_dict[cluster].append(index)
            cluster_accuracies_ordered_by_cluster = [cluster_acc_tuple[1] for cluster_acc_tuple in
                                                     sorted(ordered_clusters_accuracies, key = lambda cluster_acc_tuple: cluster_acc_tuple[0])]
            cluster_accuracies_row_indices = [(x[1], x[0][1]) for x in sorted(
                zip(cluster_row_indices_dict.items(), cluster_accuracies_ordered_by_cluster),
                key = lambda cluster_acc_tuple: cluster_acc_tuple[1], reverse = True)]

            accuracy_row_indices_dict = {acc: [] for acc in cluster_accuracies_ordered_by_cluster}
            for acc_row_indices_tuple in cluster_accuracies_row_indices:
                accuracy_row_indices_dict[acc_row_indices_tuple[0]] += acc_row_indices_tuple[1]

            alg_group_subsample_indices = []
            remaining_subsample_size = subsample_size
            remaining_accuracy_bins = [acc_bin for acc_bin in accuracy_bins]
            remaining_accuracy_subsample_sizes = {acc_bin: size for acc_bin, size in accuracy_subsample_sizes.items()}
            loop_remaining_accuracy_bins = [acc_bin for acc_bin in remaining_accuracy_bins]
            while remaining_subsample_size > 0:
                for acc_bin in loop_remaining_accuracy_bins:
                    if acc_bin not in accuracy_row_indices_dict:
                        residual = remaining_accuracy_subsample_sizes[acc_bin]
                        remaining_accuracy_bins.remove(acc_bin)
                        remaining_accuracy_subsample_sizes[acc_bin] = 0
                        remaining_accuracy_subsample_sizes = redistribute_residual_subsample(
                            residual, remaining_accuracy_bins, accuracy_subsample_weights, remaining_accuracy_subsample_sizes)
                    elif remaining_accuracy_subsample_sizes[acc_bin] > len(accuracy_row_indices_dict[acc_bin]):
                        acc_bin_subsample_size = len(accuracy_row_indices_dict[acc_bin])
                        remaining_subsample_size -= acc_bin_subsample_size
                        residual = remaining_accuracy_subsample_sizes[acc_bin] - acc_bin_subsample_size
                        remaining_accuracy_bins.remove(acc_bin)
                        remaining_accuracy_subsample_sizes[acc_bin] = 0
                        alg_group_subsample_indices += accuracy_row_indices_dict[acc_bin]
                        remaining_accuracy_subsample_sizes = redistribute_residual_subsample(
                            residual, remaining_accuracy_bins, accuracy_subsample_weights, remaining_accuracy_subsample_sizes)
                    else:
                        alg_group_subsample_indices += np.random.choice(
                            accuracy_row_indices_dict[acc_bin], remaining_accuracy_subsample_sizes[acc_bin], replace = False).tolist()
                        remaining_subsample_size -= remaining_accuracy_subsample_sizes[acc_bin]
                        remaining_accuracy_subsample_sizes[acc_bin] = 0
                loop_remaining_accuracy_bins = remaining_accuracy_bins

            for i in alg_group_subsample_indices:
                subsample_row_indices.append(tuple(multiindex_list + [i])) 

        else:
            for i in alg_group_df.index:
                subsample_row_indices.append(tuple(multiindex_list + [i]))

    subsampled_df = prediction_df_orig.loc[sorted(subsample_row_indices)]

    return subsampled_df

def redistribute_residual_subsample(residual, remaining_accuracy_bins, accuracy_subsample_weights, remaining_accuracy_subsample_sizes):

    residual_of_residual = residual
    sum_remaining_weights = 0
    for acc_bin in remaining_accuracy_bins:
        sum_remaining_weights += accuracy_subsample_weights[acc_bin]
    for acc_bin in remaining_accuracy_bins:
        acc_bin_redistributed_residual = int(residual * accuracy_subsample_weights[acc_bin] / sum_remaining_weights)
        remaining_accuracy_subsample_sizes[acc_bin] += acc_bin_redistributed_residual
        residual_of_residual -= acc_bin_redistributed_residual
    remaining_accuracy_subsample_sizes[remaining_accuracy_bins[0]] += residual_of_residual

    return remaining_accuracy_subsample_sizes

def update_training_data(prediction_df):

    try:
        training_df, = load_pkl_objects(training_dir, 'training_df_test')
        training_df = pd.concat([training_df, prediction_df])
    except FileNotFoundError:
        training_df = prediction_df
    save_pkl_objects(training_dir, **{'training_df_test': training_df})

    prediction_df_csv = prediction_df.copy()
    prediction_df_csv['timestamp'] = str(datetime.datetime.now()).split('.')[0]
    prediction_df_csv.reset_index(inplace = True)
    try:
        training_df_csv = pd.read_csv(join(training_dir, 'training_df_test.csv'))
        training_df_csv = pd.concat([training_df_csv, prediction_df_csv])
    except FileNotFoundError:
        training_df_csv = prediction_df_csv
    training_df_csv.set_index(['timestamp', 'scan'], inplace = True)
    training_df_csv.to_csv(join(training_dir, 'training_df_test.csv'))

    return training_df

def make_training_forests(training_df, alg_list, cores):

    train_target_arr_dict = make_train_target_arr_dict(training_df, alg_list)
    
    optimized_params = optimize_model(train_target_arr_dict, cores)

    forest_dict = make_forest_dict(train_target_arr_dict, optimized_params, cores)

    save_pkl_objects(training_dir, **{'forest_dict': forest_dict})
    #forest_dict, = load_pkl_objects(training_dir, 'forest_dict')
    
    return forest_dict

def optimize_model(train_target_arr_dict, cores):

    cores = 3

    optimized_params = {}
    for alg_key in train_target_arr_dict:
        optimized_params[alg_key] = {}

        data_train_split, data_validation_split, target_train_split, target_validation_split =\
            train_test_split(train_target_arr_dict[alg_key]['train'], train_target_arr_dict[alg_key]['target'], stratify = train_target_arr_dict[alg_key]['target'])
        forest_grid = GridSearchCV(RandomForestClassifier(n_estimators = 150, oob_score = True),
                              {'max_features': ['sqrt', None], 'max_depth': [depth for depth in range(11, 20)]},
                              n_jobs = cores)
        forest_grid.fit(data_train_split, target_train_split)
        optimized_forest = forest_grid.best_estimator_
        optimized_params[alg_key] = optimized_forest.max_depth
        optimized_params[alg_key] = optimized_forest.max_features

        #plot_feature_importances(optimized_forest, alg_key, train_target_arr_dict[alg_key]['feature_names'])
        plot_errors(data_train_split, data_validation_split, target_train_split, target_validation_split, alg_key, cores)

    return optimized_params

def plot_feature_importances(forest, alg_key, feature_names):

    importances = forest.feature_importances_
    feature_std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis = 0)
    indices = np.argsort(importances)[::-1]
    
    fig, ax = plt.subplots()
    ax.set_title('Feature importances')
    x = np.arange(len(importances))
    ax.bar(left = x, height = importances[indices], color = 'r', yerr = feature_std[indices], width = 0.9, align = 'center')
    ax.set_xticks(x)
    labels = np.array(feature_names)[indices]
    ax.set_xticklabels(labels, rotation = -45, ha = 'left')
    ax.set_xlim([-1, len(importances)])
    ax.set_ylim(ymin = 0)
    fig.set_tight_layout(True)

    alg_key_str = '_'.join(alg_key)
    save_path = join(test_dir, alg_key_str + '_feature_importances.png')
    fig.savefig(save_path, bbox_inches = 'tight')

def plot_errors(data_train_split, data_validation_split, target_train_split, target_validation_split, alg_key, cores):

    cores = 3

    ensemble_clfs = [
        ('max_features=\'sqrt\'',
         RandomForestClassifier(warm_start = True, max_features = 'sqrt', oob_score = True, max_depth = 15, n_jobs = cores, random_state = 1)),
        ('max_features=None',
         RandomForestClassifier(warm_start = True, max_features = None, oob_score = True, max_depth = 15, n_jobs = cores, random_state = 1))
    ]

    oob_errors = OrderedDict((label, []) for label, _ in ensemble_clfs)
    validation_errors = OrderedDict((label, []) for label, _ in ensemble_clfs)
    min_estimators = 10
    max_estimators = 500

    for label, clf in ensemble_clfs:
        for tree_number in range(min_estimators, max_estimators + 1, 50):
            clf.set_params(n_estimators = tree_number)
            clf.fit(data_train_split, target_train_split)

            oob_error = 1 - clf.oob_score_
            oob_errors[label].append((tree_number, oob_error))

            validation_error = 1 - clf.score(data_validation_split, target_validation_split)
            validation_errors[label].append((tree_number, validation_error))

    fig, ax1 = plt.subplots()
    for label, oob_error in oob_errors.items():
        xs, ys = zip(*oob_error)
        ax1.plot(xs, ys, label = 'oob error: ' + label)
    for label, validation_error in validation_errors.items():
        xs, ys = zip(*validation_error)
        ax1.plot(xs, ys, label = 'validation error: ' + label)

    ax1.set_xlim(min_estimators, max_estimators)
    ax1.set_xlabel('n_estimators')
    ax1.set_ylabel('error rate')
    ax1.legend(loc = 'upper right')
    fig.set_tight_layout(True)

    alg_key_str = '_'.join(alg_key)
    save_path = join(test_dir, alg_key_str + '_' + label + '_error.png')
    fig.savefig(save_path, bbox_inches = 'tight')

def make_forest_dict(train_target_arr_dict, optimized_params, cores):

    cores = 3

    forest_dict = {}.fromkeys(train_target_arr_dict)
    for alg_key in forest_dict:
        train_data = train_target_arr_dict[alg_key]['train']
        target_data = train_target_arr_dict[alg_key]['target']
        forest = RandomForestClassifier(n_estimators = n_estimators,
                                        max_depth = optimized_params['max_depth'],
                                        max_features = optimized_params['max_features'],
                                        oob_score = True,
                                        n_jobs = cores)
        forest.fit(train_data, target_data)
        forest_dict[alg_key] = forest

    return forest_dict

def standardize_prediction_df_cols(prediction_df):

    for accepted_mass_tol in accepted_mass_tols:
        if accepted_mass_tol not in prediction_df.columns:
            prediction_df[accepted_mass_tol] = 0
    prediction_df.drop('is top rank single alg', inplace = True)
    min_retention_time = prediction_df['retention time'].min()
    max_retention_time = prediction_df['retention time'].max()
    prediction_df['retention time'] = (prediction_df['retention time'] - min_retention_time) / (max_retention_time - min_retention_time)
    prediction_df.sort_index(1, inplace = True)
    return prediction_df

def make_train_target_arr_dict(training_df, alg_list):

    training_df.sort_index(inplace = True)
    multiindex_groups = list(product((0, 1), repeat = len(alg_list)))[1:]
    model_keys_used = []
    train_target_arr_dict = {}
    for multiindex in multiindex_groups:
        model_key = tuple([alg for i, alg in enumerate(alg_list) if multiindex[i]])
        model_keys_used.append(model_key)
        train_target_arr_dict[model_keys_used[-1]] = {}.fromkeys(['train', 'target'])
        try:
            alg_group_df = training_df.xs(multiindex).reset_index().set_index(['scan', 'seq'])
            alg_group_df.dropna(1, inplace = True)
            train_columns = alg_group_df.columns.tolist()
            train_columns.remove('ref match')
            train_target_arr_dict[model_key]['train'] = alg_group_df.as_matrix(train_columns)
            train_target_arr_dict[model_key]['target'] = alg_group_df['ref match'].tolist()
            train_target_arr_dict[model_key]['feature_names'] = train_columns
        except KeyError:
            print(str(model_keys_used[-1]) + ' predictions were not found')

    return train_target_arr_dict

def find_target_accuracy(prediction_df, ref_file, cores):

    ref = load_ref(ref_file)
    single_var_match_seq = partial(match_seq_to_ref, ref = ref)

    ### TEMPORARY ###
    cores = 3
    #################

    if cores == 1:
        grouped_by_seq = prediction_df.groupby('seq')['seq']
        prediction_df['ref match'] = grouped_by_seq.transform(single_var_match_seq)

    else:
        partitioned_prediction_df_list = []
        partitioned_grouped_by_seq_series_list = []
        total_rows = len(prediction_df)
        first_row_number_split = int(total_rows / cores)

        row_number_splits = [(0, first_row_number_split)]
        for core_number in range(2, cores):
            row_number_splits.append((row_number_splits[-1][1], first_row_number_split * core_number))

        row_number_splits.append(
            (row_number_splits[-1][1], total_rows))

        for row_number_split in row_number_splits:
            partitioned_prediction_df_list.append(prediction_df.ix[row_number_split[0]: row_number_split[1]])
            partitioned_grouped_by_seq_series_list.append(partitioned_prediction_df_list[-1].groupby('seq')['seq'])

        multiprocessing_pool = Pool(cores)
        multiprocessing_match_seqs_to_ref_for_partitioned_group =\
            partial(match_seqs_to_ref_for_partitioned_group, single_var_match_seq_fn = single_var_match_seq)
        partitioned_seq_match_series_list = multiprocessing_pool.map(
            multiprocessing_match_seqs_to_ref_for_partitioned_group, partitioned_grouped_by_seq_series_list)
        multiprocessing_pool.close()
        multiprocessing_pool.join()

        prediction_df['ref match'] = pd.concat(partitioned_seq_match_series_list)

    return prediction_df

def match_seq_to_ref(grouped_by_seq_series, ref):

    query_seq = grouped_by_seq_series.iloc[0]
    print(query_seq)
    for target_seq in ref:
        if query_seq in target_seq:
            return 1
    return 0

def match_seqs_to_ref_for_partitioned_group(partitioned_grouped_by_seq_series, single_var_match_seq_fn):

    partitioned_seq_match_series = partitioned_grouped_by_seq_series.transform(single_var_match_seq_fn)

    return partitioned_seq_match_series

def load_ref(ref_file):

    with open(ref_file) as f:
        lines = f.readlines()

    ref = []
    for line in lines:
        if line[0] == '>':
            next_seq_in_next_line = True
        elif line != '\n':
            if next_seq_in_next_line:
                ref.append(line.strip().replace('I', 'L'))
                next_seq_in_next_line = False
            else:
                ref[-1] += line.strip().replace('I', 'L')

    return ref