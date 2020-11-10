# Source code for multi-label image classificaiton models

We are going to show the steps to run the experiments in batch.
You can feed the input to the model one by one, but it is not effective.


## Step 0: Dependencies
1. opencv-python for image processing, use `pip3 install opencv-python`
2. [pycoco](https://github.com/cocodataset/cocoapi), we need it to extract data from COCO annotation file.
3. [ASL](https://github.com/Alibaba-MIIL/ASL) if you want to test the ASL pre-trained model on COCO

Please change the import in `ASL_MLC_testing` so that the import of ASL model works.

## Step 1: Download COCO Validation Set

You can find the instruction from COCO website.

We need both the image and the annotations files. 

We assume the image is in `./cocodataset/val2014` and the annotations file is in `./cocodataset/annotations`

Besides, please put a copy of image in `./coco/org`, using softlink

## Step 1: Mutate the image

Run the `mutate_coco.py` 
It will mutate all images in `./coco/org`, and put them into some folders like `./coco/bg_255`.
You can control the number of colors to be used to mutate in the source code. 

## Step 3: Download the pre-trained models

For asl model, please refet to (here)[https://github.com/Alibaba-MIIL/ASL/blob/main/MODEL_ZOO.md]
For resnet model, please refer to (here)[https://github.com/ARiSE-Lab/DeepInspect/blob/master/deepinspect/coco/model_best.pth.tar]

## Step 4: Feed the inputs to models

if you want to test the ResNet50 model used in our paper, please try `resnet_mlc_testing`

if you want to test the ASL model used in our paper, please try `ASL_mlc_testing`

## Step 5: Analyze the data

Please see `analyze_result.py`
