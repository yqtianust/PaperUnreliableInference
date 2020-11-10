"""
Provides code for keras pretrained models evaluation
"""

from __future__ import absolute_import

import numpy as np
import os, csv

from .utils import create_folder


def analyze():
    """Evaluate accuracy, neuron coverage and robustness of all pretrained
    models.
    Top-1 accuracy and top-5 accuracy are both measured.
    For neuron coverage, measured thresholds include 0.0, 0.1, 0.2, ..., 0.9.
    To measure robustness, three adversarial attack methods, FGSM, BIM and
    DeepFoolAttack, are used.
    """
    models =  ['xception', 'vgg16', 'vgg19', 'resnet50',
               'resnet101', 'resnet152', 'resnet50_v2', 'resnet101_v2',
            'resnet152_v2',  'inception_v3', 'inception_resnet_v2','mobilenet',
               'mobilenet_v2', 'densenet121', 'densenet169', 'densenet201',
            'nasnet_mobile', 'nasnet_large']

    img_dirs = ['./img/org', './img/bg2_0', './img/bg2_127', './img/bg2_255', './img/obj2_0', './img/obj2_127',
                './img/obj2_255']
    bg_img_dirs = img_dirs[1:4]
    obj_img_dirs = img_dirs[4:]


    csvfile = open("./summary_slc.csv", 'w')

    fieldnames = ['model_name', 'acc', 'MR1', 'MR1_acc', 'MR1_correct', 'MR1_incorrect',
                  'non_MR1', 'non_MR1_acc', 'non_MR1_correct', 'non_MR1_incorrect',
                  'MR2', 'MR2_acc', 'MR2_correct', 'MR2_incorrect',
                  'non_MR2', 'non_MR2_acc', 'non_MR2_correct', 'non_MR2_incorrect',
                  'MR1&2', 'MR1&2_acc', 'MR1&2_correct', 'MR1&2_incorrect',
                  'non_MR1&2', 'non_MR1&2_acc', 'non_MR1&2_correct', 'non_MR1&2_incorrect', 'acc_num']
    csvwriter = csv.DictWriter(csvfile, fieldnames)
    csvwriter.writeheader()

    for model_name in models[1:2]:

        # obj is removed
        org = np.load("./npz_file/{}_{}_all_probs.npz".format(model_name, "./img/org".replace("/", "-")),
                      allow_pickle=True)

        org_y_ture = org['y_ture']
        org_y_pred = org['y_pred']
        org_y_argsort = np.argsort(org_y_pred, axis=1)
        org_sorted = np.take_along_axis(org_y_pred, org_y_argsort, axis=1)
        org_label = org_y_argsort[:, -1]
        org_certain = org_sorted[:, -1] - org_sorted[:, -2]

        ground_truth_acc = org_y_ture == org_label

        # bg is removed

        all_bg_acc = np.zeros([len(bg_img_dirs), 50000])
        for i, img_dir in enumerate(bg_img_dirs):
            bg_data = np.load("./npz_file/{}_{}_all_probs.npz".format(model_name, img_dir.replace("/", "-")),
                              allow_pickle=True)
            bg_y_pred = bg_data['y_pred'], 1000
            bg_argsort = np.argsort(bg_y_pred, axis=1)
            # sorted = np.take_along_axis(bg_y_pred, bg_argsort, axis=1)
            all_bg_acc[i, :] = org_y_ture == bg_argsort[:, -1]

        result_bg = np.zeros([50000])
        for i in range(0, 50000):
            # if True:
            arr = all_bg_acc[:, i]
            if np.sum(arr) < 2:
                result_bg[i] = True

        all_obj_label = np.zeros([len(obj_img_dirs), 50000]).astype(np.int32)
        all_obj_certain = np.zeros([len(obj_img_dirs), 50000])
        for i, img_dir in enumerate(obj_img_dirs):
            obj_data = np.load("./npz_file/{}_{}_all_probs.npz".format(model_name, img_dir.replace("/", "-")),
                               allow_pickle=True)
            obj_y_pred = obj_data['y_pred']
            obj_argsort = np.argsort(obj_y_pred, axis=1)
            sorted = np.take_along_axis(obj_y_pred, obj_argsort, axis=1)
            all_obj_label[i, :] = obj_argsort[:, -1]
            all_obj_certain[i, :] = sorted[:, -1] - sorted[:, -2]

        result_obj = np.zeros([50000])

        for i in range(0, 50000):
            acc_arr = all_obj_label[:, i] == org_label[i]
            certain_arr = org_certain[i] > all_obj_certain[:, i]
            mr_1_result = np.zeros_like(acc_arr)
            for j in range(0, len(acc_arr)):
                if acc_arr[j] == False or (acc_arr[j] == True and certain_arr[j] == True):
                    mr_1_result[j] = True

            if np.sum(mr_1_result[-3:]) < 2:
                result_obj[i] = True


        common = np.logical_and(result_bg, result_obj)


        total = 50000
        count_obj = np.sum(result_obj)
        count_bg = np.sum(result_bg)
        count_intersection = np.sum(common)

        csvwriter.writerow({'model_name':model_name, 'acc':"{:.3%}".format(np.sum(ground_truth_acc)/total),
        'MR1': count_obj, 'MR1_acc': "{:.3%}".format(np.sum(np.logical_and(result_obj, ground_truth_acc))/count_obj),
        'MR1_correct': np.sum(np.logical_and(result_obj, ground_truth_acc)),
        'MR1_incorrect': np.sum(np.logical_and(result_obj, np.logical_not(ground_truth_acc))),
        'non_MR1': total - count_obj,
        'non_MR1_acc': "{:.3%}".format(np.sum(np.logical_and(np.logical_not(result_obj), ground_truth_acc)) /(total- count_obj)),
        'non_MR1_correct': np.sum(np.logical_and(np.logical_not(result_obj), ground_truth_acc)),
        'non_MR1_incorrect': np.sum(np.logical_and(np.logical_not(result_obj), np.logical_not(ground_truth_acc))),

        'MR2':count_bg, 'MR2_acc':"{:.3%}".format(np.sum(np.logical_and(result_bg, ground_truth_acc))/count_bg),
        'MR2_correct':np.sum(np.logical_and(result_bg, ground_truth_acc)),
        'MR2_incorrect':np.sum(np.logical_and(result_bg, np.logical_not(ground_truth_acc))),
        'non_MR2': total - count_bg,
        'non_MR2_acc': "{:.3%}".format(np.sum(np.logical_and(np.logical_not(result_bg), ground_truth_acc)) /(total- count_bg)),
        'non_MR2_correct': np.sum(np.logical_and(np.logical_not(result_bg), ground_truth_acc)),
        'non_MR2_incorrect': np.sum(np.logical_and(np.logical_not(result_bg), np.logical_not(ground_truth_acc))),

        'MR1&2':count_intersection, 'MR1&2_acc':"{:.3%}".format(np.sum(np.logical_and(common, ground_truth_acc))/count_intersection),
        'MR1&2_correct':np.sum(np.logical_and(common, ground_truth_acc)),
        'MR1&2_incorrect':np.sum(np.logical_and(common, np.logical_not(ground_truth_acc))),
        'non_MR1&2': total - count_intersection,
        'non_MR1&2_acc': "{:.3%}".format(np.sum(np.logical_and(np.logical_not(common), ground_truth_acc)) /(total- count_intersection)),
        'non_MR1&2_correct': np.sum(np.logical_and(np.logical_not(common), ground_truth_acc)),
        'non_MR1&2_incorrect': np.sum(np.logical_and(np.logical_not(common), np.logical_not(ground_truth_acc))),
        'acc_num': np.sum(ground_truth_acc)
        })

        np.savez(os.path.join("./all_probs_analyze_distribution", "{}.npz".format(model_name)), acc=ground_truth_acc,
                 violate_obj=np.logical_and(result_obj, ground_truth_acc),
                 violate_bg=np.logical_and(result_bg, ground_truth_acc),
                 violate_both=np.logical_and(common, ground_truth_acc))


if __name__ == '__main__':

    create_folder("all_probs_analyze_distribution")
    analyze()
