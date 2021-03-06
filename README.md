## Pruning_PJ3
**CIFAR10 related code from Stanford CS231n homework http://cs231n.github.io/**

**PASCAL VOC related code from https://github.com/xuzheyuan624/yolov3-pytorch**

Thanks for taking the class. In project 3, you will work on network pruning. 
In the cifar10 folder, training code template is provided for magnitutde-based weight pruning.
You don't have to follow the code and you are welcome to make changes to the provided code for your purpose.

### CIFAR10
Change to cifar10 directory
```
cd cifar10
```
Download cifar10 dataset with script
```
sh get_datasets.sh
```
Test you can train the simple example model
```
python3 train.py
```
For loading trained checkpoint, please find following line at the end of train.py and change ckpt_dir to your own one.
```
m.construct_model(ckpt_dir=None)
```
If no error occurs, you can make your model in model.py and start working on project 3. Pruning funcions can be implemented in prune_utils.py

**Notice** If you are using Tensorflow official models at https://github.com/tensorflow/models/tree/master/research/slim/nets, please check how it can be instantiated correctly. For example, you are required to use code as follows for mobilenet_v1.
```
with tf.contrib.slim.arg_scope(mobilenet_v1_arg_scope(is_training=True)):
    out, _ = mobilenet_v1(X, num_classes=10)
```

### PASCAL VOC
**Code from https://github.com/xuzheyuan624/yolov3-pytorch**

Change to pascal_voc directory
```
cd pascal_voc
```
Create "weights" directory and download pretrained backbone weights for darknet53 to "weights" folder from https://drive.google.com/file/d/1zoGUFk9Tfoll0vCteSJRGa9I1yEAfU-a/view. 
```
cd data
sh get_voc_dataset.sh
python voc_label.py
```
Change to "data" direcotry and download pascal voc dataset. Uncompress the tar file and you should find "VOCdevkit" under "data" directory. Meanwhile, check image path names in xx_val.txt and xx_train.txt to make sure training scripts can find them.
```
cd ..
```
Return to pascal_voc direcotry and run following commands below for training/evaluation. No need to load darknet53 backbone weights with "--load" argument, since it is automatically loaded every time you run the script. Use "--load" to optionally specify the path for your own checkpoint. 
```
python3 main.py train --load PRETRAINED_PTH --name=voc --gpu False
python3 main.py eval --load PRETRAINED_PTH --name=voc --gpu False
```
Pruning funcions can be implemented in prune_utils.py. 

**Notice** Evaluation can be very slow for not-well-trained model due to too many predicted bounding boxes. So train several epoches then evaluate.

**Notice** I have uploaded pretrained weights for yolov3 on PASCAL VOC on google drive https://drive.google.com/file/d/1PnhVkGkjiBalNK_gBNS0bw9SN39eLcXu/view?usp=sharing. It has been trained for 27 epoches and achieves  mAP(IoU=0.5) as 73.2, which can be used as a starting point.
