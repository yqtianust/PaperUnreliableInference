"""
Provides code for keras pretrained models evaluation
"""

from __future__ import absolute_import

import numpy as np
import os, csv
import pickle

# from utils import create_folder

def compare_l(a, b):
    if a.shape != b.shape:
        return False
    else:
        return (a==b).all()

def larger_p(a, b):
    if a.shape != b.shape:
        return False
    else:
        return (a>b).all()


def analyze():
    """Evaluate accuracy, neuron coverage and robustness of all pretrained
    models.
    Top-1 accuracy and top-5 accuracy are both measured.
    For neuron coverage, measured thresholds include 0.0, 0.1, 0.2, ..., 0.9.
    To measure robustness, three adversarial attack methods, FGSM, BIM and
    DeepFoolAttack, are used.
    """
    models = ['L', 'XL', 'MLCCOCO']

    # img_dirs = ['./img/org']
    # img_dirs = ['./img/bg_0', './img/bg_127', './img/bg_255', './img/obj_0', './img/obj_127', './img/obj_255']
    # img_dirs = ['./img/bg2_0', './img/bg2_127', './img/bg2_255', './img/obj2_0', './img/obj2_127', './img/obj2_255']
    # gt_path = './ILSVRC2012_validation_ground_truth.txt'

    # result_dict = {}
    csvfile = open("./summary_mlccoco.csv", 'w')

    fieldnames = ['model_name', 'acc', 'MR1', 'MR1_acc', 'MR1_correct', 'MR1_incorrect',
                  'non_MR1', 'non_MR1_acc', 'non_MR1_correct', 'non_MR1_incorrect',
                  'MR2', 'MR2_acc', 'MR2_correct', 'MR2_incorrect',
                  'non_MR2', 'non_MR2_acc', 'non_MR2_correct', 'non_MR2_incorrect',
                  'MR1&2', 'MR1&2_acc', 'MR1&2_correct', 'MR1&2_incorrect',
                  'non_MR1&2', 'non_MR1&2_acc', 'non_MR1&2_correct', 'non_MR1&2_incorrect', 'acc_num']
    csvwriter = csv.DictWriter(csvfile, fieldnames)
    csvwriter.writeheader()

    for model_name in models[0:]:

        threshold = 0.5 if model_name is 'MLCCOCO' else 0.7

        # TODO ground_truth acc
        # TODO need to know what is average precision....
        # if ((target_bool[i] == pred_sample[i]).all()):
        gt = pickle.load(open("COCO_ground_truth.pickle", "rb"))

        org = pickle.load(open("./result_pickle/{}_{}.pickle".format(model_name, "org".replace("/", "-")), "rb"))

        ground_truth_acc = np.zeros([len(org.keys()), 1])
        list = sorted(gt.keys())

        for i in range(0, len(list)):
            key = list[i]
            y_gt = gt[key]
            y_pred = org[key]['prob'] > threshold
            if (y_gt == y_pred).all():
                ground_truth_acc[i] = True


        # bg is removed
        bg0 = pickle.load(
            open("./result_pickle/{}_{}.pickle".format(model_name, "merged_bg2_0".replace("/", "-")), "rb"))
        bg127 = pickle.load(
            open("./result_pickle/{}_{}.pickle".format(model_name, "merged_bg2_127".replace("/", "-")), "rb"))
        bg255 = pickle.load(
            open("./result_pickle/{}_{}.pickle".format(model_name, "merged_bg2_255".replace("/", "-")), "rb"))


        result_bg = np.zeros([len(org.keys()), 1])

        for i in range(0, len(list)):
            key = list[i]
            if True:
                org_l = org[key]['prob'] > threshold

                if key not in bg0.keys():
                    continue

                bg0_l = bg0[key]['prob'] > threshold
                bg127_l = bg127[key]['prob'] > threshold
                bg255_l = bg255[key]['prob'] > threshold

                arr = [bg0_l, bg127_l, bg255_l]
                arr = [compare_l(ele, org_l) for ele in arr]
                if np.sum(arr) < 2:
                    result_bg[i] = True

        # obj is removed
        obj0 = pickle.load(
            open("./result_pickle/{}_{}.pickle".format(model_name, "merged_obj2_0".replace("/", "-")), "rb"))
        obj127 = pickle.load(
            open("./result_pickle/{}_{}.pickle".format(model_name, "merged_obj2_127".replace("/", "-")), "rb"))
        obj255 = pickle.load(
            open("./result_pickle/{}_{}.pickle".format(model_name, "merged_obj2_255".replace("/", "-")), "rb"))



        result_obj = np.zeros([len(org.keys()), 1])

        for i in range(0, len(list)):
            key = list[i]
            if True:
                org_l = org[key]['prob'] > threshold
                if key not in obj0.keys():
                    continue
                obj0_l = obj0[key]['prob'] > threshold
                obj127_l = obj127[key]['prob'] > threshold
                obj255_l = obj255[key]['prob'] > threshold

                org_probs = org[key]['prob'][org_l]
                obj0_probs = obj0[key]['prob'][obj0_l]
                obj127_probs = obj127[key]['prob'][obj127_l]
                obj255_probs = obj255[key]['prob'][obj255_l]

                individual_count = 0

                if (not compare_l(org_l, obj0_l)) or (
                        compare_l(org_l, obj0_l) and larger_p(org_probs, obj0_probs)):
                    individual_count += 1

                if (not compare_l(org_l, obj127_l)) or (
                        compare_l(org_l, obj127_l) and larger_p(org_probs, obj127_probs)):
                    individual_count += 1

                if (not compare_l(org_l, obj255_l)) or (
                        compare_l(org_l, obj255_l) and larger_p(org_probs, obj255_probs)):
                    individual_count += 1

                if individual_count < 2:
                    result_obj[i] = True

        common = np.logical_and(result_bg, result_obj)

        total = 40504
        count_obj = np.sum(result_obj)
        count_bg = np.sum(result_bg)
        count_intersection = np.sum(common)


        csvwriter.writerow({'model_name': model_name, 'acc': "{:.3%}".format(np.sum(ground_truth_acc) / total),
                            'MR1': count_obj, 'MR1_acc': "{:.3%}".format(
                np.sum(np.logical_and(result_obj, ground_truth_acc)) / count_obj),
                            'MR1_correct': np.sum(np.logical_and(result_obj, ground_truth_acc)),
                            'MR1_incorrect': np.sum(np.logical_and(result_obj, np.logical_not(ground_truth_acc))),
                            'non_MR1': total - count_obj,
                            'non_MR1_acc': "{:.3%}".format(
                                np.sum(np.logical_and(np.logical_not(result_obj), ground_truth_acc)) / (
                                            total - count_obj)),
                            'non_MR1_correct': np.sum(np.logical_and(np.logical_not(result_obj), ground_truth_acc)),
                            'non_MR1_incorrect': np.sum(
                                np.logical_and(np.logical_not(result_obj), np.logical_not(ground_truth_acc))),

                            'MR2': count_bg,
                            'MR2_acc': "{:.3%}".format(np.sum(np.logical_and(result_bg, ground_truth_acc)) / count_bg),
                            'MR2_correct': np.sum(np.logical_and(result_bg, ground_truth_acc)),
                            'MR2_incorrect': np.sum(np.logical_and(result_bg, np.logical_not(ground_truth_acc))),
                            'non_MR2': total - count_bg,
                            'non_MR2_acc': "{:.3%}".format(
                                np.sum(np.logical_and(np.logical_not(result_bg), ground_truth_acc)) / (
                                            total - count_bg)),
                            'non_MR2_correct': np.sum(np.logical_and(np.logical_not(result_bg), ground_truth_acc)),
                            'non_MR2_incorrect': np.sum(
                                np.logical_and(np.logical_not(result_bg), np.logical_not(ground_truth_acc))),

                            'MR1&2': count_intersection, 'MR1&2_acc': "{:.3%}".format(
                np.sum(np.logical_and(common, ground_truth_acc)) / count_intersection),
                            'MR1&2_correct': np.sum(np.logical_and(common, ground_truth_acc)),
                            'MR1&2_incorrect': np.sum(np.logical_and(common, np.logical_not(ground_truth_acc))),
                            'non_MR1&2': total - count_intersection,
                            'non_MR1&2_acc': "{:.3%}".format(
                                np.sum(np.logical_and(np.logical_not(common), ground_truth_acc)) / (
                                            total - count_intersection)),
                            'non_MR1&2_correct': np.sum(np.logical_and(np.logical_not(common), ground_truth_acc)),
                            'non_MR1&2_incorrect': np.sum(
                                np.logical_and(np.logical_not(common), np.logical_not(ground_truth_acc))),
                            'acc_num': np.sum(ground_truth_acc)
                            })

        np.savez(os.path.join("./mlc_all_probs_analyze_distribution", "{}.npz".format(model_name)), acc=ground_truth_acc,
                 violate_obj=np.logical_and(result_obj, ground_truth_acc),
                 violate_bg=np.logical_and(result_bg, ground_truth_acc),
                 violate_both=np.logical_and(common, ground_truth_acc))

if __name__ == '__main__':
    analyze()
