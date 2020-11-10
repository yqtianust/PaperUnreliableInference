import numpy as np
import torch
from .src.models.tresnet import TResNet

def infer_batch(model, classes_list, inputs, threshold=0.7):
    # inputs: batch, channel, height, weight
    # print('ASL Example Inference code on a batch of images')

    output = torch.sigmoid(model(inputs))

    probs = output.cpu().detach().numpy()
    labels = []
    labels_probs = []

    # print(type(classes_list))
    # numpy array

    for i in range(0, inputs.shape[0]):
        np_output = probs[i, :]
        # print(np_output.shape)
        detected_classes = classes_list[np_output > threshold]
        # print(detected_classes)
        labels.append(detected_classes)
        labels_probs.append(np_output[np_output > threshold])

    return probs, labels, labels_probs


def load_model(model_type):

    if model_type is "L":
        model_name = "tresnet_l"
        path = './pth_files/MS_COCO_TRresNet_L_448_86.6.pth'
        input_size = 448
        threshold = 0.5
    elif model_type is "XL":
        model_name = "tresnet_xl"
        path = './pth_files/MS_COCO_TResNet_xl_640_88.4.pth'
        input_size = 640
        threshold = 0.5

    state = torch.load(path, map_location='cpu')
    num_classes = state['num_classes']

    if model_type is "L":
        do_bottleneck_head = False
        model = TResNet(layers=[4, 5, 18, 3], num_classes=num_classes, in_chans=3, width_factor=1.2,
                        do_bottleneck_head=do_bottleneck_head)
    elif model_type is "XL":
        model = TResNet(layers=[4, 5, 24, 3], num_classes=num_classes, in_chans=3, width_factor=1.3)

    model = model.cuda()
    model.load_state_dict(state['model'], strict=True)
    model.eval()

    classes_list = np.array(list(state['idx_to_class'].values()))

    return model, input_size, threshold, num_classes, classes_list