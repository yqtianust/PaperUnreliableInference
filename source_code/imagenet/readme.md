# Source code for single-label image classificaiton models

We are going to show the steps to run the experiments in batch.
In other words, we will mutate all images and then feed them into models.
It is possible to mutate one and 

## Step 0: Dependencies
1. opencv-python for image processing, use `pip3 install opencv-python`
2. [EvalDNN](https://github.com/yqtianust/EvalDNN).
3. Keras. 

## Step 1: Download ImageNet Validation Set

You can find the instruction from imagenet website.

We need both the image and the annotations (xml file).

We assume the image is in `./img/org` and the xml file is in `./val2012_xml`

## Step 2: Mutate the image

Run the `mutate_img.py`. 
It will mutate all images in `./img/org`, and put them into some folders like `./img/bg_RED`.
You can control the number of colors to be used to mutate in the source code. 

## Step 3: Download the pre-trained models

If you are using keras, the models should be downloaded automatically. 

## Step 4: Feed the inputs to models

Run the `run_keras.py`

You can control the number of mutated images, and the number of models to be tested in the source code. 


## Step 5: Analyze the data

`analyze_result.py`

