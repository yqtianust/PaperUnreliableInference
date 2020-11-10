import os
from collections import OrderedDict
import pickle
from tqdm import tqdm as tqdm

import torch
from torchvision.transforms import transforms

from .resnet_model import MultilabelObject
from .test_data_loader import COCO_MLC
from .mlc_utils import infer_batch


def main():

    # img_folders = ["../coco_img/bg2_0/", "../coco_img/bg2_127/", "../coco_img/bg2_255/",
                   # "../coco_img/obj2_0/", "../coco_img/obj2_127/", "../coco_img/obj2_255/"]

    img_folders = ["../coco_img/merged_bg2_0/", "../coco_img/merged_bg2_127/", "../coco_img/merged_bg2_255/",
                   "../coco_img/merged_obj2_0/", "../coco_img/merged_obj2_127/", "../coco_img/merged_obj2_255/"]
    img_folders = ["../coco_img/org/"]

    model_name = "MLCCOCO"

    model = MultilabelObject(None, 80).cuda()

    log_dir = "./"
    checkpoint = torch.load(os.path.join(log_dir, 'model_best.pth.tar'), encoding='bytes')
    new_checkpoint = OrderedDict()
    for k in checkpoint[b'state_dict']:
        new_checkpoint[k.decode('utf-8')] = checkpoint[b'state_dict'][k]

    model.load_state_dict(new_checkpoint)
    model.eval()

    with open("classes_list.pickle", "rb") as f:
        classes_list = pickle.load(f)

    for img_folder in img_folders:

        crop_size = 224
        image_size = 256
        batch_size = 64
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        val_transform = transforms.Compose([
            transforms.Scale(image_size),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize])

        cocomlc = COCO_MLC(img_folder, val_transform)

        test_loader = torch.utils.data.DataLoader(cocomlc, batch_size=batch_size,
                                                  shuffle=False, num_workers=8,
                                                  pin_memory=True)


        t = tqdm(test_loader, desc='testing {}'.format(img_folder))

        result = {}

        for batch_idx, (imgs, paths) in enumerate(t):

            images = imgs.cuda()
            # print(images.shape)

            probs, labels, labels_probs = infer_batch(model, classes_list, inputs=images, threshold=0.5)

            for i in range(0, len(paths)):
                path = paths[i]
                result[path] = {"prob": probs[i], "labels": labels[i], "labels_probs": labels_probs[i]}

        pickle_file_name = "{}_{}.pickle".format(model_name, os.path.basename(os.path.normpath(img_folder)))
        pickle_path = os.path.join(".", "result_pickle", pickle_file_name)
        with open(pickle_path, 'wb') as handle:
            pickle.dump(result, handle)
        print("Done, Saved to {}".format(pickle_path))


if __name__ == '__main__':
    main()
