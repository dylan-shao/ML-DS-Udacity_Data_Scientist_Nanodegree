## Train the data

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
> example:
```
python train.py 'flowers/train' --save_dir 'mydir' --arch 'vgg16' --learning_rate 0.001 --epochs 5
```


## Predict the data

 ```python predict.py /path/to/image checkpoint```

 * Options:
     * Return top K most likely classes: python predict.py input checkpoint --top_k 3
 > example:
 ```python predict.py 'flowers/test/1/image_06743.jpg' 'checkpoints/checkpoint.pth' --top_k 5```
