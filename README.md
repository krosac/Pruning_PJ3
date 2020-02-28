## Pruning_PJ3
Thanks for taking the class. In project 3, you will work on network pruning. 
In the cifar10 folder, training code template is provided for magnitutde-based weight pruning.
You don't have to follow the code and you are welcome to make changes to the provided code for your purpose.

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
If no error occurs, you can make your model in model.py and start working on project 3. Puning funcions can be implemented in prune_utils.py


