## Train the data

> example:
```
python train.py 'flowers/train' --save_dir 'mydir' --arch 'vgg16' --learning_rate 0.001 --epochs 5 --gpu True
```

### Basic Usage:

inside the folder, call
```
python train.py data_directory

```

### Additional args

* Options:
    * Set directory to save checkpoints: ```--save_dir "save_directory"```
    * Choose architecture: ```--arch "vgg13"```
    * Set hyperparameters: ```--learning_rate 0.001 --epochs 20```
    * Using gpu to train, default is cpu ```--gpu True```



## Predict the data

> example:
```
python predict.py 'flowers/test/1/image_06743.jpg' 'checkpoints/checkpoint.pth' --top_k 5 --gpu True --file  my_cat_to_name.json
```

### Basic Usage:

 ```python predict.py /path/to/image checkpoint```

 example: ```python predict.py 'flowers/test/1/image_06743.jpg'```

### Additional args

 * Options:
     * Return top K most likely classes: ```--top_k 3```
     * Using gpu to predict ```--gpu True```
     * using customized mapping ```--file my_cat_to_name.json```, this will load this json file and use this instead of the default one
