import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--file', type=argparse.FileType('r'))
args = parser.parse_args()


with args.file as file:
    a = json.load(file)
    print(a['test'])
