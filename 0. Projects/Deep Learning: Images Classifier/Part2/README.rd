# Train the data

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


# Predict the data
