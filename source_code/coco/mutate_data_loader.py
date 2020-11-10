import json, os, string, random, time, pickle, gc
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms

from pycocotools.coco import COCO

class CocoObject(data.Dataset):
    def __init__(self, ann_dir, image_dir, split = 'train', transform = None):
        self.ann_dir = ann_dir
        self.image_dir = image_dir
        self.split = split
        self.transform = transform

        if self.split == 'train':
            ann_path = os.path.join(self.ann_dir, "instances_train2014.json")
        else:
            ann_path = os.path.join(self.ann_dir, "instances_val2014.json")
        self.cocoAPI = COCO(ann_path)
        self.data = json.load(open(ann_path))
        self.image_ids = [elem['id'] for elem in self.data['images']]

        if self.split == 'val':
            self.image_ids = self.image_ids
        # elif self.split == 'test':
        #     self.image_ids = self.image_ids[10000:]

        print("Num of image ids {}".format(len(self.image_ids)))
        self.image_path_map = {elem['id']: elem['file_name'] for elem in self.data['images']}
        #80 objects
        id2object = dict()
        object2id = dict()
        for idx, elem in enumerate(self.data['categories']):
            id2object[idx] = elem['name']
            object2id[elem['name']] = idx

        self.id2object = id2object
        self.object2id = object2id
        with open('id2object.pickle', 'wb') as handle:
            pickle.dump(id2object, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open('object2id.pickle', 'wb') as handle:
            pickle.dump(object2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
        self.id2labels = {}
        #generate one-hot encoding objects annotation for every image
        self.object_ann = np.zeros((len(self.image_ids), 80))

        # self.ann_cat_id = {}
        # self.ann_cat_name = {}
        #
        # self.bbox = {}
        self.mask = {}


        for idx, image_id in enumerate(self.image_ids):
            ann_ids = self.cocoAPI.getAnnIds(imgIds = image_id)
            anns = self.cocoAPI.loadAnns(ids = ann_ids)
            category_ids = [elem['category_id'] for elem in anns]
            category_names = [elem['name'] for elem in self.cocoAPI.loadCats(ids=category_ids)]
            self.id2labels[image_id] = category_names
            encoding_ids = [object2id[name] for name in category_names]
            for encoding_id in encoding_ids:
                self.object_ann[idx, encoding_id] = 1


            # self.ann_cat_name[image_id] = category_names
            # ann_catid = [object2id[elem] for elem in category_names]
            # bboxes = [elem['bbox'] for elem in anns]
            # bboxes = [[round(i) for i in bbox] for bbox in bboxes]
            # masks = [self.cocoAPI.annToMask(ann) for ann in anns]
            # self.ann_cat_id[image_id] = ann_catid
            # self.bbox[image_id] = bboxes
            # self.mask[image_id] = masks

        # img_object_ann = {}
        # for idx, image_id in enumerate(self.image_ids):
        #     img_object_ann[self.image_path_map[image_id]] = self.object_ann[idx]
        # with open("COCO_ground_truth.pickle", "wb") as handle:
        #     pickle.dump(img_object_ann, handle)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_file_name = self.image_path_map[image_id]
        if self.split == 'train':
            image_path = os.path.join(self.image_dir,"train2014", image_file_name)
        else:
            image_path = os.path.join(self.image_dir,"val2014", image_file_name)

        img = Image.open(image_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.Tensor(self.object_ann[index]), image_id

    def getObjectWeights(self):
        return (self.object_ann == 0).sum(axis = 0) / (1e-8 + self.object_ann.sum(axis = 0))

    def __len__(self):
        return len(self.image_ids)