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
    * Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
    * Choose architecture: python train.py data_dir --arch "vgg13"
    * Set hyperparameters: python train.py data_dir --learning_rate 0.001 --epochs 20



## Predict the data

> example:
```
python predict.py 'flowers/test/1/image_06743.jpg' 'checkpoints/checkpoint.pth' --top_k 5 --gpu True --file  my_cat_to_name.json
```

 ```python predict.py /path/to/image checkpoint```

 * Options:
     * Return top K most likely classes: python predict.py input checkpoint --top_k 3
     * Using gpu to predict ```--gpu True```
     * using customized mapping ```--file  my_cat_to_name.json```, this will load this json file and use this instead of the default one
