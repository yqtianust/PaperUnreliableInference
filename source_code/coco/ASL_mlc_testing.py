import pickle
import os
from tqdm import tqdm as tqdm

import torch
from torchvision.transforms import transforms

from .mlc_utils import infer_batch, load_model
from .test_data_loader import COCO_MLC


def main():

    img_folders = ["../coco_img/org/",
                   "../coco_img/bg_0/", "../coco_img/bg_127/", "../coco_img/bg_255/",
                   "../coco_img/obj_0/", "../coco_img/obj_127/", "../coco_img/obj_255/"]


    for model_name in ["L", "XL"]:

        # if model_name is "L":
        #     continue

        for img_folder in img_folders:

            model, input_size, threshold, num_classes, classes_list = load_model(model_name)
            if model_name is "L":
                batch_size = 12
            else:
                batch_size = 4

            val_transform = transforms.Compose([
                transforms.Resize([input_size, input_size]),
                # transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
            ])

            # Data samplers.
            # img_folder = "../coco_img/bg_0/"
            cocomlc = COCO_MLC(img_folder, val_transform)

            test_loader = torch.utils.data.DataLoader(cocomlc, batch_size=batch_size,
                                                      shuffle=False, num_workers=6,
                                                      pin_memory=True)

            t = tqdm(test_loader, desc='testing {}'.format(img_folder))

            result = {}

            for batch_idx, (imgs, paths) in enumerate(t):

                images = imgs.cuda()

                probs, labels, labels_probs = infer_batch(model, classes_list, inputs=images)

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
