
import argparse
from argparse import RawTextHelpFormatter

from prospr.train import train
from prospr.prediction import predict
from prospr.evaluate import evaluate

desc_usg = '''ProSPr requires a multiple sequence alignment in a3m format as input.
These can be created using a local install of HHBlits or using the online server (https://toolkit.tuebingen.mpg.de/tools/hhblits, go to "Query MSA" tab in output and select "Download Full A3M")'''

parser = argparse.ArgumentParser(description=desc_usg, usage=argparse.SUPPRESS, formatter_class=RawTextHelpFormatter)
subparsers = parser.add_subparsers(help='Whether to predict distance matrices for new structures, evaluate trained networks on a test dataset, or train a network from scratch.')

predict_parser = subparsers.add_parser('predict')
predict_parser.add_argument('--a3m', help='If mode == predict, you must provide a multiple sequence alignment file in a3m format') # TODO: add code to download alignment database and use hhblits to get a3m, enabling user to provide a protein sequence in FASTA format
predict_parser.add_argument('-o', '--output_dir', help='Output save directory for prediction pkl.', default='./data/predictions')
predict_parser.add_argument('-n', '--network', help='ProSPr network(s) used to make prediction: all (default), a, b, or c', default='all')
predict_parser.add_argument('--device', help='The index for the desired CUDA device', default=0)
predict_parser.add_argument('--save', help='Whether to save prediction results', action='store_true')
predict_parser.set_defaults(func=predict)

eval_parser = subparsers.add_parser('evaluate')
eval_parser.add_argument('--a3m', help='If mode == evaluate, you must provide a multiple sequence alignment file in a3m format and a .pdb target file')
eval_parser.add_argument('--pdb', help='If mode == evaluate, you must provide a multiple sequence alignment file in a3m format and a .pdb target file')
eval_parser.add_argument('-o', '--output_dir', help='Output save directory for evaluation plots.', default='./data/results')
eval_parser.add_argument('-n', '--network', help='ProSPr network(s) used to make prediction: all (default), a, b, or c', default='all')
eval_parser.add_argument('--device', help='The index for the desired CUDA device', default=0)
eval_parser.set_defaults(func=evaluate)
eval_parser.set_defaults(save=False)

train_parser = subparsers.add_parser('train')
train_parser.add_argument('--crop_size', help='The sizes of crops to train on', default=64)
train_parser.add_argument('--model_name', help='The name of the saved model', default='prospr_a')
train_parser.add_argument('--device', help='The index for the desired CUDA device', default=0)
train_parser.set_defaults(func=train)

args = parser.parse_args()

if __name__ == "__main__":
    
    args = parser.parse_args()
    args.func(args)