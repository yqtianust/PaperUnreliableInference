"""
Provides code for keras pretrained models evaluation
"""

from __future__ import absolute_import

from evaldnn.metrics.accuracy import Accuracy
from evaldnn.models.keras import KerasModel
from evaldnn.utils.keras import *
from .all_probs_logger import all_probs_logger


def eval():
    """Evaluate accuracy, neuron coverage and robustness of all pretrained
    models.
    Top-1 accuracy and top-5 accuracy are both measured.
    For neuron coverage, measured thresholds include 0.0, 0.1, 0.2, ..., 0.9.
    To measure robustness, three adversarial attack methods, FGSM, BIM and
    DeepFoolAttack, are used.
    """
    models =  ['vgg16', 'vgg19', 'resnet50', 'resnet101',
            'resnet152', 'resnet50_v2', 'resnet101_v2',
            'resnet152_v2', 'mobilenet', 'mobilenet_v2',
            'inception_resnet_v2', 'inception_v3', 'xception',
            'densenet121', 'densenet169', 'densenet201',
            'nasnet_mobile', 'nasnet_large']

    # img_dirs = ["./img/obj_LIME", "./img/obj_PURPLE", "./img/obj_YELLOW", "./img/obj_FUCHSIA", "./img/obj_MAROON",
    #             "./img/obj_RED", "./img/obj_AQUA", "./img/obj_GREEN", "./img/obj_OLIVE", "./img/obj_TEAL",
    #             "./img/obj_NAVY", "./img/obj_BLUE",
    #             "./img/bg_BLUE", "./img/bg_LIME", "./img/bg_PURPLE", "./img/bg_YELLOW", "./img/bg_OLIVE",
    #             "./img/bg_TEAL", "./img/bg_AQUA", "./img/bg_GREEN", "./img/bg_NAVY", "./img/bg_FUCHSIA",
    #             "./img/bg_MAROON", "./img/bg_RED" ]

    img_dirs = ['./img/org',
                './img/bg_0', './img/bg_127', './img/bg_255',
                './img/obj_0', './img/obj_127', './img/obj_255']

    gt_path = './ILSVRC2012_validation_ground_truth.txt'

    for img_dir in img_dirs[12:13]:
        for model_name in models[0:1]:

            print('Model: {}, dataset {}'.format(model_name, img_dir))

            # load the model and data
            model, data_normalized, data_original, mean, std, flip_axis, bounds = imagenet_benchmark_zoo(model_name, data_original_shuffle=False, img_dir = img_dir, ground_truth_path=gt_path, data_original_num_max=50000)

            # wrap the model with EvalDNN
            measure_model = KerasModel(model)
            logging = all_probs_logger()

            # evaluate the top-1 and top-5 accuracy of the model
            # accuracy = Accuracy()
            measure_model.predict(data_normalized.x, data_normalized.y, [logging.update])
            print("Saving to {}".format(model_name, img_dir.replace("/", "-")))

            logging.save("{}_{}_all_probs".format(model_name, img_dir.replace("/", "-")))


if __name__ == '__main__':
    eval()