# TODO:
# create gpu input to train on gpu, and default as cpu

import argparse
import ImageClassifier as IC

def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="The path of the trainning data folder")
    parser.add_argument("--save_dir", help="checkpoint director")
    parser.add_argument("--arch", help="model architecture")
    parser.add_argument("-lr","--learning_rate", help="learning rate", type=float)
    parser.add_argument("--epochs", help="epochs", type=int)
    parser.add_argument("--gpu", help="train on gpu", type=bool)

    args = parser.parse_args()
    save_dir = args.save_dir + '/' or ''
    arch = args.arch or 'vgg16'
    learning_rate = args.learning_rate or 0.001
    epochs = args.epochs or 3

    model = IC.ImageClassifier(args.path, save_dir, arch, learning_rate, epochs, args.gpu)
    model.train()

if __name__ == '__main__':
    Main()
