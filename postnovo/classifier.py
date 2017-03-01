from config import *

import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import sys
import time
import datetime
import warnings
warnings.filterwarnings('ignore')
import os
import os.path

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
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from os.path import join, basename
from multiprocessing import Pool, current_process
from collections import Counter
from scipy.stats import norm

seq_matching_count = 0

def classify(ref_file, alg_list, prediction_df = None):
    verbose_print()

    if _run_type[0] in ['train', 'test', 'optimize']:
        prediction_df = find_target_accuracy(prediction_df, ref_file)

    verbose_print('formatting data for compatability with model')
    prediction_df = standardize_prediction_df_cols(prediction_df)
    save_pkl_objects(test_dir, **{'prediction_df': prediction_df})
    #prediction_df, = load_pkl_objects(test_dir, 'prediction_df')

    if _run_type[0] in ['train', 'optimize']:
        
        subsampled_df = subsample_training_data(prediction_df, alg_list)
        save_pkl_objects(test_dir, **{'subsampled_df': subsampled_df})
        #subsampled_df, = load_pkl_objects(test_dir, 'subsampled_df')

        verbose_print('updating training database')
        training_df = update_training_data(subsampled_df)
        #training_df, = load_pkl_objects(training_dir, 'training_df')

        forest_dict = make_training_forests(training_df, alg_list)
        save_pkl_objects(training_dir, **{'forest_dict': forest_dict})
        #forest_dict, = load_pkl_objects(training_dir, 'forest_dict')

    elif _run_type[0] in ['predict', 'test']:
        
        reported_prediction_df = make_predictions(prediction_df, alg_list)
        reported_prediction_df.to_csv(os.path.join(output_dir, 'best_predictions.csv'))

def make_predictions(prediction_df, alg_list):

    forest_dict, = load_pkl_objects(training_dir, 'forest_dict')

    prediction_df['probability'] = np.nan
    alg_group_multiindex_keys = list(product((0, 1), repeat = len(alg_list)))[1:]
    for multiindex_key in alg_group_multiindex_keys:
        alg_group = tuple([alg for i, alg in enumerate(alg_list) if multiindex_key[i]])

        alg_group_data = prediction_df.xs(multiindex_key)
        accuracy_labels = alg_group_data['ref match'].tolist()
        alg_group_data.drop(['seq', 'ref match', 'probability'], axis = 1, inplace = True)
        alg_group_data.dropna(1, inplace = True)
        forest_dict[alg_group].n_jobs = _cores[0]
        probabilities = forest_dict[alg_group].predict_proba(alg_group_data.as_matrix())[:, 1]

        if _run_type[0] == 'test':
            verbose_print('making', '_'.join(alg_group), 'test plots')
            plot_roc_curve(accuracy_labels, probabilities, alg_group, alg_group_data)
            plot_precision_recall_curve(accuracy_labels, probabilities, alg_group, alg_group_data)

        prediction_df.loc[multiindex_key, 'probability'] = probabilities

    if _run_type[0] == 'test':
        plot_precision_yield(alg_group_multiindex_keys, prediction_df, alg_list)

    max_probabilities = prediction_df.groupby(prediction_df.index.get_level_values('scan'))['probability'].transform(max)
    best_prediction_df = prediction_df[prediction_df['probability'] == max_probabilities]
    reported_prediction_df = best_prediction_df[best_prediction_df['probability'] >= _min_prob[0]]
    reported_prediction_df.reset_index(inplace = True)
    reported_prediction_df.set_index('scan', inplace = True)
    
    reported_cols_in_order = []
    for reported_df_col in reported_df_cols:
        if reported_df_col in reported_prediction_df.columns:
            reported_cols_in_order.append(reported_df_col)
    reported_prediction_df = reported_prediction_df.reindex_axis(reported_cols_in_order, axis = 1)

    return reported_prediction_df

def plot_precision_yield(alg_group_multiindex_keys, prediction_df, alg_list):

    fig, ax = plt.subplots()
    plt.title('precision vs sequence yield')
    x_min = 1
    x_max = 0
    plt.ylim([0, 1])
    plt.xlabel('sequence yield')
    plt.ylabel('precision = ' + r'$\frac{T_p}{T_p + F_p}$')

    for multiindex_key in alg_group_multiindex_keys:

        fig, ax = plt.subplots()
        plt.title('precision vs sequence yield')
        x_min = 1
        x_max = 0
        plt.ylim([0, 1])
        plt.xlabel('sequence yield')
        plt.ylabel('precision = ' + r'$\frac{T_p}{T_p + F_p}$')

        alg_group = tuple([alg for i, alg in enumerate(alg_list) if multiindex_key[i]])

        alg_group_data = prediction_df.xs(multiindex_key)
        sample_size_list = list(range(1, len(alg_group_data) + 1))

        precision = make_precision_col(alg_group_data, 'probability')
        alg_group_data['random forest precision'] = precision

        x = sample_size_list[::100]
        y = precision[::100]
        z = (alg_group_data['probability'].argsort() / alg_group_data['probability'].size).tolist()[::100]
        model_line_collection = colorline(x, y, z)
        plt.colorbar(model_line_collection, label = 'moving threshold:\nrandom forest probability or\nde novo algorithm score percentile')
        annotation_x = x[len(x) // 2]
        annotation_y = y[len(y) // 2]
        plt.annotate('_'.join(alg_group) + '\n' + 'random forest',
                     xy = (annotation_x, annotation_y),
                     xycoords = 'data',
                     xytext = (25, 25),
                     textcoords = 'offset pixels',
                     arrowprops = dict(facecolor = 'black', shrink = 0.01, width = 1, headwidth = 6),
                     horizontalalignment = 'left', verticalalignment = 'bottom',
                     )

        if x[-1] > x_max:
            x_max = x[-1]
            plt.xlim([x_min, x_max])

        arrow_position = 2.5
        for i, alg in enumerate(alg_group):
            if alg == 'novor':
                score_str = 'avg novor aa score'
            elif alg == 'pn':
                score_str = 'rank score'
            precision = make_precision_col(alg_group_data, score_str)
            alg_group_data[alg + ' precision'] = precision

            x = sample_size_list[::100]
            y = precision[::100]
            z = (alg_group_data[score_str].argsort() / alg_group_data[score_str].size).tolist()[::100]
            colorline(x, y, z)
            annotation_x = x[int(len(x) / arrow_position)]
            annotation_y = y[int(len(y) / arrow_position)]
            arrow_position -= 1
            plt.annotate('_'.join(alg_group) + '\n' + score_str,
                         xy = (annotation_x, annotation_y),
                         xycoords = 'data',
                         xytext = (-25, -25),
                         textcoords = 'offset pixels',
                         arrowprops = dict(facecolor = 'black', shrink = 0.01, width = 1, headwidth = 6),
                         horizontalalignment = 'right', verticalalignment = 'top',
                         )

            if x[-1] > x_max:
                x_max = x[-1]
                plt.xlim([x_min, x_max])

        plt.tight_layout(True)
        save_path = join(test_dir, '_'.join(alg_group) + '_precision_yield.pdf')
        fig.savefig(save_path, bbox_inches = 'tight')
        plt.close()
        
    #plt.tight_layout(True)
    #save_path = join(test_dir, 'all_precision_yield.pdf')
    #fig.savefig(save_path, bbox_inches = 'tight')

    sys.exit(0)

def make_precision_col(df, sort_col):
    df.sort_values(sort_col, ascending = False, inplace = True)
    ref_matches = df['ref match']
    cumulative_ref_matches = [ref_matches.iat[0]]
    precision = [cumulative_ref_matches[0] / 1]
    for i, ref_match in enumerate(ref_matches[1:]):
        cumulative_ref_match_sum = cumulative_ref_matches[-1] + ref_match
        cumulative_ref_matches.append(cumulative_ref_match_sum)
        precision.append(cumulative_ref_match_sum / (i + 2))

    return precision

def plot_precision_recall_curve(accuracy_labels, probabilities, alg_group, alg_group_data):

    true_positive_rate, recall, thresholds = precision_recall_curve(accuracy_labels, probabilities, pos_label = 1)
    model_auc = average_precision_score(accuracy_labels, probabilities)

    alg_scores_dict = {}
    for alg in alg_group:
        if alg == 'novor':
            alg_scores_dict[alg] = alg_group_data['avg novor aa score']
        elif alg == 'pn':
            alg_scores_dict[alg] = alg_group_data['rank score']

    alg_pr_dict = {}
    alg_auc_dict = {}
    for alg in alg_scores_dict:
        alg_pr_dict[alg] = precision_recall_curve(accuracy_labels, alg_scores_dict[alg], pos_label = 1)
        alg_auc_dict[alg] = average_precision_score(accuracy_labels, alg_scores_dict[alg])

    fig, ax = plt.subplots()

    model_line_collection = colorline(recall, true_positive_rate, thresholds)
    plt.colorbar(model_line_collection, label = 'moving threshold:\nrandom forest probability or\nde novo algorithm score percentile')
    annotation_x = recall[int(len(recall) / 1.2)]
    annotation_y = true_positive_rate[int(len(true_positive_rate) / 1.2)]
    plt.annotate('random forest\nauc = ' + str(round(model_auc, 2)),
                 xy = (annotation_x, annotation_y),
                 xycoords = 'data',
                 xytext = (annotation_x + 25, annotation_y + 25),
                 textcoords = 'offset pixels',
                 arrowprops = dict(facecolor = 'black', shrink = 0.01, width = 1, headwidth = 6),
                 horizontalalignment = 'right', verticalalignment = 'bottom',
                 )

    arrow_position = 1.2
    for alg in alg_pr_dict:
        alg_recall = alg_pr_dict[alg][1]
        alg_tpr = alg_pr_dict[alg][0]
        alg_thresh = alg_pr_dict[alg][2].argsort() / alg_pr_dict[alg][2].size
        annotation_x = alg_recall[int(len(alg_recall) / arrow_position)]
        annotation_y = alg_tpr[int(len(alg_tpr) / arrow_position)]
        arrow_position -= 1
        colorline(alg_recall, alg_tpr, alg_thresh)
        plt.annotate(alg + '\nauc = ' + str(round(alg_auc_dict[alg], 2)),
                     xy = (annotation_x, annotation_y),
                     xycoords = 'data',
                     xytext = (annotation_x - 25, annotation_y - 25),
                     textcoords = 'offset pixels',
                     arrowprops = dict(facecolor = 'black', shrink = 0.01, width = 1, headwidth = 6),
                     horizontalalignment = 'right', verticalalignment = 'top',
                     )

    if len(alg_group) == 1:
        plt.title('precision-recall curve: ' + alg_group[0] + ' sequences')
    else:
        plt.title('precision-recall curve: ' + '-'.join(alg_group) + ' consensus sequences')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('recall (true positive rate) = ' + r'$\frac{T_p}{T_p + F_n}$')
    plt.ylabel('precision = ' + r'$\frac{T_p}{T_p + F_p}$')
    plt.tight_layout(True)

    save_path = join(test_dir, '_'.join(alg_group) + '_precision_recall.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')


def plot_roc_curve(accuracy_labels, probabilities, alg_group, alg_group_data):

    false_positive_rate, true_positive_rate, thresholds = roc_curve(accuracy_labels, probabilities, pos_label = 1)
    model_auc = roc_auc_score(accuracy_labels, probabilities)

    alg_scores_dict = {}
    for alg in alg_group:
        if alg == 'novor':
            alg_scores_dict[alg] = alg_group_data['avg novor aa score']
        elif alg == 'pn':
            alg_scores_dict[alg] = alg_group_data['rank score']

    alg_roc_dict = {}
    alg_auc_dict = {}
    for alg in alg_scores_dict:
        alg_roc_dict[alg] = roc_curve(accuracy_labels, alg_scores_dict[alg], pos_label = 1)
        alg_auc_dict[alg] = roc_auc_score(accuracy_labels, alg_scores_dict[alg])

    fig, ax = plt.subplots()

    model_line_collection = colorline(false_positive_rate, true_positive_rate, thresholds)
    plt.colorbar(model_line_collection, label = 'moving threshold:\nrandom forest probability or\nde novo algorithm score percentile')
    annotation_x = false_positive_rate[len(false_positive_rate) // 2]
    annotation_y = true_positive_rate[len(true_positive_rate) // 2]
    plt.annotate('random forest\nauc = ' + str(round(model_auc, 2)),
                 xy = (annotation_x, annotation_y),
                 xycoords='data',
                 xytext = (annotation_x + 50, annotation_y - 50),
                 textcoords = 'offset pixels',
                 arrowprops = dict(facecolor = 'black', shrink = 0.01, width = 1, headwidth = 6),
                 horizontalalignment = 'left', verticalalignment = 'top',
                 )

    arrow_position = 3
    for alg in alg_roc_dict:
        alg_fpr = alg_roc_dict[alg][0]
        alg_tpr = alg_roc_dict[alg][1]
        alg_thresh = alg_roc_dict[alg][2].argsort() / alg_roc_dict[alg][2].size
        annotation_x = alg_fpr[len(alg_fpr) // arrow_position]
        annotation_y = alg_tpr[len(alg_tpr) // arrow_position]
        arrow_position -= 1
        colorline(alg_fpr, alg_tpr, alg_thresh)
        plt.annotate(alg + '\nauc = ' + str(round(alg_auc_dict[alg], 2)),
                     xy = (annotation_x, annotation_y),
                     xycoords='data',
                     xytext = (annotation_x + 50, annotation_y - 50),
                     textcoords = 'offset pixels',
                     arrowprops = dict(facecolor = 'black', shrink = 0.01, width = 1, headwidth = 6),
                     horizontalalignment = 'left', verticalalignment = 'top',
                     )

    plt.plot([0, 1], [0, 1], linestyle = '--', c = 'black')

    if len(alg_group) == 1:
        plt.title('roc curve: ' + alg_group[0] + ' sequences')
    else:
        plt.title('roc curve: ' + '-'.join(alg_group) + ' consensus sequences')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('false positive rate = ' + r'$\frac{F_p}{F_p + T_n}$')
    plt.ylabel('true positive rate = ' + r'$\frac{T_p}{T_p + F_n}$')
    plt.tight_layout(True)

    save_path = join(test_dir, '_'.join(alg_group) + '_roc.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')

def colorline(x, y, z, cmap = 'jet', norm = plt.Normalize(0.0, 1.0), linewidth = 3, alpha = 1.0):

    z = np.asarray(z)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    line_collection = mcoll.LineCollection(segments, array = z, cmap = cmap, norm = norm, linewidth = linewidth, alpha = alpha)

    ax = plt.gca()
    ax.add_collection(line_collection)
    return line_collection

def subsample_training_data(prediction_df_orig, alg_list):

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

def make_training_forests(training_df, alg_list):

    train_target_arr_dict = make_train_target_arr_dict(training_df, alg_list)
    
    if _run_type[0] == 'train':
        forest_dict = make_forest_dict(train_target_arr_dict)
    elif _run_type[0] == 'optimize':
        verbose_print('optimizing random forest parameters')
        optimized_params = optimize_model(train_target_arr_dict)
        forest_dict = make_forest_dict(train_target_arr_dict, optimized_params)

    return forest_dict

def optimize_model(train_target_arr_dict):

    optimized_params = {}
    for alg_key in train_target_arr_dict:
        optimized_params[alg_key] = {}

        data_train_split, data_validation_split, target_train_split, target_validation_split =\
            train_test_split(train_target_arr_dict[alg_key]['train'], train_target_arr_dict[alg_key]['target'], stratify = train_target_arr_dict[alg_key]['target'])
        forest_grid = GridSearchCV(RandomForestClassifier(n_estimators = 150, oob_score = True),
                              {'max_features': ['sqrt', None], 'max_depth': [depth for depth in range(11, 20)]},
                              n_jobs = _cores[0])
        forest_grid.fit(data_train_split, target_train_split)
        optimized_forest = forest_grid.best_estimator_
        optimized_params[alg_key]['max_depth'] = optimized_forest.max_depth
        verbose_print(alg_key, 'optimized max depth:', optimized_forest.max_depth)
        optimized_params[alg_key]['max_features'] = optimized_forest.max_features
        verbose_print(alg_key, 'optimized max features:', optimized_forest.max_features)

        plot_feature_importances(optimized_forest, alg_key, train_target_arr_dict[alg_key]['feature_names'])
        plot_errors(data_train_split, data_validation_split, target_train_split, target_validation_split, alg_key)

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
    save_path = join(test_dir, alg_key_str + '_feature_importances.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')

def plot_errors(data_train_split, data_validation_split, target_train_split, target_validation_split, alg_key):
    if len(alg_key) > 1:
        verbose_print('plotting errors vs tree size for', '-'.join(alg_key), 'consensus sequences')
    else:
        verbose_print('plotting errors vs tree size for', alg_key[0], 'sequences')

    ensemble_clfs = [
        ('max_features=\'sqrt\'',
         RandomForestClassifier(warm_start = True, max_features = 'sqrt', oob_score = True, max_depth = 15, n_jobs = _cores[0], random_state = 1)),
        ('max_features=None',
         RandomForestClassifier(warm_start = True, max_features = None, oob_score = True, max_depth = 15, n_jobs = _cores[0], random_state = 1))
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
    save_path = join(test_dir, alg_key_str + '_error.pdf')
    fig.savefig(save_path, bbox_inches = 'tight')

def make_forest_dict(train_target_arr_dict, optimized_params = default_optimized_params):

    forest_dict = {}.fromkeys(train_target_arr_dict)
    for alg_key in forest_dict:
        if len(alg_key) > 1:
            verbose_print('making random forest for', '-'.join(alg_key), 'consensus sequences')
        else:
            verbose_print('making random forest for', alg_key[0], 'sequences')

        train_data = train_target_arr_dict[alg_key]['train']
        target_data = train_target_arr_dict[alg_key]['target']
        forest = RandomForestClassifier(n_estimators = n_estimators,
                                        max_depth = optimized_params[alg_key]['max_depth'],
                                        max_features = optimized_params[alg_key]['max_features'],
                                        oob_score = True,
                                        n_jobs = _cores[0])
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

def find_target_accuracy(prediction_df, ref_file):
    verbose_print('loading', basename(ref_file))
    ref = load_ref(ref_file)
    verbose_print('finding sequence matches to reffile')

    seq_set_list = list(set(prediction_df['seq']))
    one_percent_number_seqs = len(seq_set_list) / 100 / _cores[0]

    multiprocessing_pool = Pool(_cores[0])
    single_var_match_seq = partial(match_seq_to_ref, ref = ref, one_percent_number_seqs = one_percent_number_seqs, cores = _cores[0])
    seq_set_matches = multiprocessing_pool.map(single_var_match_seq, seq_set_list)
    multiprocessing_pool.close()
    multiprocessing_pool.join()

    seq_match_dict = dict(zip(seq_set_list, seq_set_matches))
    single_var_get_match_from_dict = partial(get_match_from_dict, seq_match_dict = seq_match_dict)
    prediction_df['ref match'] = prediction_df['seq'].apply(single_var_get_match_from_dict)

    verbose_print()
    return prediction_df

def match_seq_to_ref(query_seq, ref, one_percent_number_seqs, cores):

    if current_process()._identity[0] % cores == 1:
        global seq_matching_count
        seq_matching_count += 1
        if int(seq_matching_count % one_percent_number_seqs) == 0:
            percent_complete = int(seq_matching_count / one_percent_number_seqs)
            if percent_complete <= 100:
                verbose_print_over_same_line('reference sequence matching progress: ' + str(percent_complete) + '%')

    for target_seq in ref:
        if query_seq in target_seq:
            return 1
    return 0

def get_match_from_dict(seq, seq_match_dict):
    return seq_match_dict[seq]

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