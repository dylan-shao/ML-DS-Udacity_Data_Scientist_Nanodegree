# * python predict.py /path/to/image checkpoint
# * Options:
#     * Return top K most likely classes: python predict.py input checkpoint --top_k 3
# example:
# python predict.py 'flowers/test/1/image_06743.jpg' 'checkpoints/checkpoint.pth' --top_k 5

import argparse
import ImageClassifier as IC
import torch

def load_checkpoint(checkpoint_path):
    checkpoints = torch.load(checkpoint_path)

    # TODO: some of the variable should not be needed when init the model, refactor them into other class methods
    instance = IC.ImageClassifier(checkpoints['data_dir'],checkpoints['save_dir'], \
            checkpoints['arch'],checkpoints['learning_rate'],checkpoints['epochs'])

    instance.set_state_dict(checkpoints['state_dict'])
    return instance

def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help="The path to the image to predict")
    parser.add_argument("checkpoint_path", help="The path to the checkpoint")
    parser.add_argument("--top_k", help="top k", type=int)

    args = parser.parse_args()
    top_k = args.top_k or 10

    model = load_checkpoint(args.checkpoint_path)

    model.predict(args.image_path, top_k)

if __name__ == '__main__':
    Main()
