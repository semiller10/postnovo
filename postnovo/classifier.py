from config import *

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import time
import datetime
import warnings
warnings.filterwarnings('ignore')
import os

from utils import (save_pkl_objects, load_pkl_objects,
                   save_json_objects, load_json_objects,
                   verbose_print, verbose_print_over_same_line)

from functools import partial
from itertools import product
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.cluster import Birch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from os.path import join, basename
from multiprocessing import Pool, current_process
from collections import Counter
from scipy.stats import norm

seq_matching_count = 0

def classify(prediction_df, train, ref_file, cores, alg_list):
    verbose_print()

    if train:
        prediction_df = find_target_accuracy(prediction_df, ref_file, cores)

        verbose_print('setting up subsampling')
        prediction_df = standardize_prediction_df_cols(prediction_df)
        save_pkl_objects(test_dir, **{'prediction_df': prediction_df})
        #prediction_df, = load_pkl_objects(test_dir, 'prediction_df')

        subsampled_df = subsample_training_data(prediction_df, alg_list, cores)
        save_pkl_objects(test_dir, **{'subsampled_df': subsampled_df})
        #subsampled_df, = load_pkl_objects(test_dir, 'subsampled_df')

        verbose_print('updating training database')
        training_df = update_training_data(prediction_df)
        #training_df, = load_pkl_objects(training_dir, 'training_df')

        make_training_forests(training_df, alg_list, cores)
        save_pkl_objects(training_dir, **{'forest_dict': forest_dict})
        #forest_dict, = load_pkl_objects(training_dir, 'forest_dict')

    else:
        forest_dict, = load_pkl_objects(training_dir, 'forest_dict')
        alg_group_multiindex_keys = list(product((0, 1), repeat = len(alg_list)))[1:]
        for multiindex_key in alg_group_multiindex_keys:
            alg_group = tuple([alg for i, alg in enumerate(alg_list) if multiindex[i]])
            sample_data = prediction_df.xs(multiindex_key)
            accuracy_labels = sample_data['ref match']
            sample_data.drop('ref match', axis = 1, inplace = True)
            class_probabilities = forest_dict[alg_group].predict_proba(sample_data.as_matrix(), n_jobs = cores)
            fpr, tpr, thresholds = metrics.roc_curve(accuracy_labels, class_probabilities, pos_label = 1)

            fig, ax = plt.subplots()
            ax.set_title('_'.join(alg_group) + ' roc curve')
            ax.plot(fpr, tpr, color = 'r')
            ax.plot([0, 1], [0, 1], linestyle = '--')
            ax.set_xlabel('false positive rate')
            ax.set_ylabel('true positive rate')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            fig.set_tight_layout(True)

            save_path = join(test_dir, '_'.join(alg_group) + '_roc.png')
            fig.savefig(save_path, bbox_inches = 'tight')

def subsample_training_data(prediction_df_orig, alg_list, cores):

    subsample_row_indices = []
    train_target_arr_dict = list(product((0, 1), repeat = len(alg_list)))[1:]
    prediction_df_orig['unique index'] = [i for i in range(prediction_df_orig.shape[0])]
    prediction_df_orig.set_index('unique index', append = True, inplace = True)
    prediction_df = prediction_df_orig.copy()
    prediction_df.drop(['is top rank single alg', 'seq'], axis = 1, inplace = True)

    accuracy_bins = sorted([round(x / accuracy_divisor, 1) for x in range(accuracy_divisor)], reverse = True)
    
    lower = accuracy_distribution_lower_bound
    upper = accuracy_distribution_upper_bound
    weight_bins = np.arange(lower, upper + (upper - lower) / accuracy_divisor, (upper - lower) / accuracy_divisor)
    sigma = accuracy_distribution_sigma
    mu_location = accuracy_distribution_mu_location
    accuracy_weights = (norm.cdf(weight_bins[1: 1 + accuracy_divisor], loc = mu_location, scale = sigma)
                        - norm.cdf(weight_bins[: accuracy_divisor], loc = mu_location, scale = sigma))\
                            / (norm.cdf(upper, loc = mu_location, scale = sigma)
                               - norm.cdf(lower, loc = mu_location, scale = sigma))
    accuracy_subsample_weights = {acc_bin: weight for acc_bin, weight in zip(accuracy_bins, accuracy_weights)}
    accuracy_subsample_sizes = {acc_bin: int(weight * subsample_size) for acc_bin, weight in accuracy_subsample_weights.items()}
    while sum(accuracy_subsample_sizes.values()) != subsample_size:
        accuracy_subsample_sizes[accuracy_bins[0]] += 1

    for multiindex_key in train_target_arr_dict:
        multiindex_list = list(multiindex_key)
        alg_group_df_key = tuple([alg for i, alg in enumerate(alg_list) if multiindex_key[i]])
        if sum(multiindex_key) == 1:
            verbose_print('subsampling', alg_group_df_key[0], 'top-ranking sequences')
        else:
            verbose_print('subsampling', '-'.join(alg_group_df_key), 'consensus sequences')
        alg_group_df = prediction_df.xs(multiindex_key)
        alg_group_unique_index = alg_group_df.index.get_level_values('unique index')
        alg_group_df.reset_index(inplace = True)
        alg_group_df.set_index(['scan'], inplace = True)
        alg_group_df.dropna(1, inplace = True)
        ref_match_col = alg_group_df['ref match'].copy()

        retained_features_target = round(clustering_feature_retention_factor_dict[sum(multiindex_key)] / alg_group_df.shape[0], 0)
        if retained_features_target < min_retained_features_target:
            retained_features_target = min_retained_features_target
        retained_features_list = []
        retained_feature_count = 0
        for feature in features_ordered_by_importance:
            if feature in alg_group_df.columns:
                retained_features_list.append(feature)
                retained_feature_count += 1
            if retained_feature_count == retained_features_target:
                break
        alg_group_df = alg_group_df[retained_features_list]

        if alg_group_df.shape[0] > subsample_size:

            pipe = make_pipeline(StandardScaler(),
                                 Birch(threshold = birch_threshold, n_clusters = None))
            cluster_assignments = pipe.fit_predict(alg_group_df.as_matrix())

            cluster_assignment_accuracies = zip(cluster_assignments, ref_match_col)
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

            scan_index = alg_group_df.index
            for i in alg_group_subsample_indices:
                subsample_row_indices.append(tuple(multiindex_list + [scan_index[i], alg_group_unique_index[i]])) 

        else:
            for i, scan in enumerate(alg_group_df.index):
                subsample_row_indices.append(tuple(multiindex_list + [scan, alg_group_unique_index[i]]))

    subsampled_df = prediction_df_orig.loc[sorted(subsample_row_indices)]
    retained_multiindices = subsampled_df.index.names[:-1]
    subsampled_df.reset_index(inplace = True)
    subsampled_df.drop('unique index', axis = 1, inplace = True)
    subsampled_df.set_index(retained_multiindices, inplace = True)

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
        training_df, = load_pkl_objects(training_dir, 'training_df')
        training_df = pd.concat([training_df, prediction_df])
    except FileNotFoundError:
        training_df = prediction_df
    save_pkl_objects(training_dir, **{'training_df': training_df})

    prediction_df_csv = prediction_df.copy()
    prediction_df_csv['timestamp'] = str(datetime.datetime.now()).split('.')[0]
    prediction_df_csv.reset_index(inplace = True)
    try:
        training_df_csv = pd.read_csv(join(training_dir, 'training_df.csv'))
        training_df_csv = pd.concat([training_df_csv, prediction_df_csv])
    except FileNotFoundError:
        training_df_csv = prediction_df_csv
    training_df_csv.set_index(['timestamp', 'scan'], inplace = True)
    training_df_csv.to_csv(join(training_dir, 'training_df.csv'))

    return training_df

def make_training_forests(training_df, alg_list, cores):

    train_target_arr_dict = make_train_target_arr_dict(training_df, alg_list)
    
    verbose_print('optimizing random forest parameters')
    optimized_params = optimize_model(train_target_arr_dict, cores)

    forest_dict = make_forest_dict(train_target_arr_dict, optimized_params, cores)

    return forest_dict

def optimize_model(train_target_arr_dict, cores):

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
        optimized_params[alg_key]['max_depth'] = optimized_forest.max_depth
        optimized_params[alg_key]['max_features'] = optimized_forest.max_features

        plot_feature_importances(optimized_forest, alg_key, train_target_arr_dict[alg_key]['feature_names'])
        plot_errors(data_train_split, data_validation_split, target_train_split, target_validation_split, alg_key, cores)

    return optimized_params

def plot_feature_importances(forest, alg_key, feature_names):
    if len(alg_key) > 1:
        verbose_print('plotting feature importances for', '-'.join(alg_key), 'consensus sequences')
    else:
        verbose_print('plotting feature importances for', alg_key[0], 'sequences')

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
    if len(alg_key) > 1:
        verbose_print('plotting errors vs tree size for', '-'.join(alg_key), 'consensus sequences')
    else:
        verbose_print('plotting errors vs tree size for', alg_key[0], 'sequences')

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
    save_path = join(test_dir, alg_key_str + '_error.png')
    fig.savefig(save_path, bbox_inches = 'tight')

def make_forest_dict(train_target_arr_dict, optimized_params, cores):

    forest_dict = {}.fromkeys(train_target_arr_dict)
    for alg_key in forest_dict:
        if len(alg_key) > 1:
            verbose_print('making random forest for', '-'.join(alg_key), 'consensus sequences')
        else:
            verbose_print('making random forest for', alg_key, 'sequences')

        train_data = train_target_arr_dict[alg_key]['train']
        target_data = train_target_arr_dict[alg_key]['target']
        forest = RandomForestClassifier(n_estimators = n_estimators,
                                        max_depth = optimized_params[alg_key]['max_depth'],
                                        max_features = optimized_params[alg_key]['max_features'],
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

    verbose_print('loading', basename(ref_file))
    ref = load_ref(ref_file)
    verbose_print('finding sequence matches to reffile')

    cores = 1

    if cores == 1:
        grouped_by_seq = prediction_df.groupby('seq')['seq']
        one_percent_number_seqs = len(grouped_by_seq) / 100
        single_var_match_seq = partial(match_seq_to_ref, ref = ref, one_percent_number_seqs = one_percent_number_seqs)
        prediction_df['ref match'] = grouped_by_seq.transform(single_var_match_seq)

    else:
        partitioned_prediction_df_list = []
        partitioned_grouped_by_seq_series_list = []
        total_rows = len(prediction_df)
        first_row_number_split = int(total_rows / cores / 60)

        row_number_splits = [(0, first_row_number_split)]
        for split_multiple in range(2, cores * 60):
            row_number_splits.append((row_number_splits[-1][1], first_row_number_split * split_multiple))

        row_number_splits.append(
            (row_number_splits[-1][1], total_rows))

        total_seqs = 0
        for row_number_split in row_number_splits:
            partitioned_prediction_df_list.append(prediction_df.ix[row_number_split[0]: row_number_split[1]])
            partitioned_grouped_by_seq_series_list.append(partitioned_prediction_df_list[-1].groupby('seq')['seq'])
            total_seqs += len(partitioned_grouped_by_seq_series_list[-1])

        print('total seqs' + str(total_seqs))

        multiprocessing_pool = Pool(cores)
        one_percent_number_seqs = total_seqs / cores / 100
        single_var_match_seq = partial(match_seq_to_ref, ref = ref,
                                       one_percent_number_seqs = one_percent_number_seqs)
        multiprocessing_match_seqs_to_ref_for_partitioned_group =\
            partial(match_seqs_to_ref_for_partitioned_group, single_var_match_seq_fn = single_var_match_seq)
        partitioned_seq_match_series_list = multiprocessing_pool.map(
            multiprocessing_match_seqs_to_ref_for_partitioned_group, partitioned_grouped_by_seq_series_list)
        multiprocessing_pool.close()
        multiprocessing_pool.join()

        prediction_df['ref match'] = pd.concat(partitioned_seq_match_series_list)

    print()
    return prediction_df

def match_seq_to_ref(grouped_by_seq_series, ref, one_percent_number_seqs):
    global seq_matching_count
    #if current_process()._identity[0] == 1:
    seq_matching_count += 1
    if int(seq_matching_count % one_percent_number_seqs) == 0:
        verbose_print_over_same_line('reference sequence matching progress: ' + str(int(seq_matching_count / one_percent_number_seqs)) + '%')

    query_seq = grouped_by_seq_series.iloc[0]
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