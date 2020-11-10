from pycocotools.coco import COCO
# from PIL import Image
from .mutate_data_loader import CocoObject
import numpy as np
import os
import cv2
from tqdm import tqdm

if __name__ == '__main__':
    ann_dir = './cocodataset/annotations'
    image_dir = './cocodataset/'
    test_data = CocoObject(ann_dir=ann_dir, image_dir=image_dir,
                           split='val', transform=None)
    image_ids = test_data.image_ids
    image_path_map = test_data.image_path_map
    # 80 objects
    id2object = test_data.id2object
    id2labels = test_data.id2labels

    # print(id2labels)
    # print(id2object)
    # exit(-1)

    ann_cat_name = test_data.ann_cat_name
    ann_cat_id = test_data.ann_cat_id
    bboxes = test_data.bbox
    masks = test_data.mask

    fill_values = [0, 127, 255]

    print("start aug")
    count = 0

    t = tqdm(image_ids)

    for image_id in t:
        anns = ann_cat_id[image_id]
        bbox = bboxes[image_id]
        mask = masks[image_id]
        path = image_path_map[image_id]

        # unique_anns = list(set(anns))
        # print(unique_anns)
        # for unique_ann in unique_anns:

        output_filename = "{}.jpg".format(path[0:-4])
        # COCO_val2014_000000240972.jpg
        # mask_to_union =[]
        # print(len(anns))
        # print(len(mask))

        if len(anns) > 0:
            union_mask = np.zeros_like(mask[0])

            for i in range(0, len(anns)):
                union_mask = np.add(union_mask, mask[i])

            union_mask = union_mask > 0

            image_path = os.path.join(image_dir, "val2014", path)
            image = cv2.imread(image_path)
            for fill_value in fill_values:
                obj_image = image.copy()
                obj_image[np.nonzero(union_mask)] = fill_value
                bg_image = image.copy()
                bg_image[np.nonzero(1 - union_mask)] = fill_value

                cv2.imwrite("../coco_img/obj_{}/{}".format(fill_value, output_filename), obj_image)
                cv2.imwrite("../coco_img/bg2_{}/{}".format(fill_value, output_filename), bg_image)

            count += 1
        else:
            print("No mask: {}".format(output_filename))
        # if count >= 10:
        #     break