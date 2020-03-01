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
```
Change to "data" direcotry and download pascal voc dataset. Uncompress the tar file and you should find "VOCdevkit" under "data" directory.
```
cd ..
```
Return to pascal_voc direcotry and run following commands below for training/evaluation. No need to load darknet53 backbone weights with "--load" argument, since it is automatically loaded every time you run the script. Use "--load" to specify the path for your own checkpoint.
```
python3 main.py train --load PRETRAINED_PTH --gpu False
python3 main.py eval --load PRETRAINED_PTH --gpu False
```
Pruning funcions can be implemented in prune_utils.py. 

**Notice** Evaluation can be very slow for not-well-trained model due to too many predicted bounding boxes. So train several epoches then evaluate.

