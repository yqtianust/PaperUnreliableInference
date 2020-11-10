import os
import cv2
from handle_bbox import get_bbox
from utils import create_folder
import numpy as np
from multiprocessing import Pool

def process_img(filename):
    image = cv2.imread(filename)
    if len(image.shape) != 3:
        print(image.shape)
        return 0
    #     image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    filename_base = os.path.basename(filename)

    h, w, c = image.shape

    boxes = get_bbox(filename_base)
    # print(image.shape)
    obj_mask_B = np.zeros_like(image)
    obj_mask_G = np.zeros_like(image)
    obj_mask_R = np.zeros_like(image)

    bg_mask_B = np.ones_like(image)
    bg_mask_G = np.ones_like(image)
    bg_mask_R = np.ones_like(image)

    bg_mask_B[..., 1:] = 0
    bg_mask_G[..., 0] = 0
    bg_mask_G[..., 2] = 0
    bg_mask_R[..., 0:2] = 0

    for box in boxes:
        xmin = max(round(box.xmin_scaled * w) - 5, 0)
        ymin = max(round(box.ymin_scaled * h) - 5, 0)
        xmax = min(round(box.xmax_scaled * w) + 5, w)
        ymax = min(round(box.ymax_scaled * h) + 5, h)

        obj_mask_B[ymin:ymax, xmin:xmax, 0] = 1
        obj_mask_G[ymin:ymax, xmin:xmax, 1] = 1
        obj_mask_R[ymin:ymax, xmin:xmax, 2] = 1

        bg_mask_B[ymin:ymax, xmin:xmax, 0] = 0
        bg_mask_G[ymin:ymax, xmin:xmax, 1] = 0
        bg_mask_R[ymin:ymax, xmin:xmax, 2] = 0

    for fill_color_name in Colors.keys():
        fill_value_R, fill_value_G, fill_value_B = Colors[fill_color_name]

        obj_image = image.copy()

        obj_image[np.nonzero(obj_mask_B)] = fill_value_B
        obj_image[np.nonzero(obj_mask_G)] = fill_value_G
        obj_image[np.nonzero(obj_mask_R)] = fill_value_R

        bg_image = image.copy()

        bg_image[np.nonzero(bg_mask_B)] = fill_value_B
        bg_image[np.nonzero(bg_mask_G)] = fill_value_G
        bg_image[np.nonzero(bg_mask_R)] = fill_value_R

        cv2.imwrite("./img/obj_{}/{}".format(fill_color_name, filename_base), obj_image)
        cv2.imwrite("./img/bg_{}/{}".format(fill_color_name, filename_base), bg_image)


if __name__ == '__main__':


    MoreColors = {"NAVY": (0, 0, 128), "FUCHSIA": (255, 0, 255),
           "SILVER": (192, 192, 192), "GRAY": (128, 128, 128),
           "RED": (255, 0, 0), "MAROON": (128, 0, 0),
           "YELLOW": (255, 255, 0), "OLIVE": (128, 128, 0),
           "LIME": (0, 255, 0), "GREEN": (0, 128, 0),
           "AQUA": (0, 255, 255), "TEAL": (0, 128, 128),
           "BLUE": (0, 0, 255), "PURPLE": (128, 0, 128)}
    SimpleColor = {"0": (0, 0, 0), "127": (127, 127, 127),
                   "255": (255, 255, 255)}

    Colors = SimpleColor
    # if you want to use more colors, uncomment the following
    # Colors = MoreColors

    for fill_color_name in Colors.keys():
        create_folder("./img/obj_{}".format(fill_color_name))
        create_folder("./img/bg_{}".format(fill_color_name))

    count = 0

    files = []

    for r, d, fs in os.walk(os.path.expanduser("./img/org")):
        for f in sorted(fs):
            files.append(os.path.join(r, f))
    pool = Pool()
    pool.map(process_img, files)

    pool.close()
    pool.join()
