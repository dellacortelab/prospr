import argparse
from argparse import RawTextHelpFormatter
import torch

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from prospr.sequence import Sequence
from prospr.nn import ProsprNetwork, load_model, CUDA
from prospr.prediction import predict_domain
from prospr.io import save

desc_usg = '''ProSPr requires a multiple sequence alignment in a3m format as input.
These can be created using a local install of HHBlits or using the online server (https://toolkit.tuebingen.mpg.de/tools/hhblits, go to "Query MSA" tab in output and select "Download Full A3M")'''
#TODO: include docker info...?

parser = argparse.ArgumentParser(description=desc_usg, usage=argparse.SUPPRESS, formatter_class=RawTextHelpFormatter)
parser.add_argument('a3m', help='Multiple sequence alignment file in a3m format')
parser.add_argument('-n', '--network', help='ProSPr network(s) used to make prediction: all (default), a, b, or c', default='all')
parser.add_argument('-o', '--output', help='Output save path for prediction pkl. Default uses same location and ID as input a3m', default='')


def main(args):

    model_paths = []
    if args.network == 'all':
        model_paths = ['./nn/prospr0421_'+x+'.pt' for x in ['a','b','c']]
    elif args.network in ['a', 'b', 'c']:
        model_paths.append('./nn/prospr0421_'+args.network+'.pt')
    else:
        print('Invalid network selection!')
        return 

    seq = Sequence(args.a3m)

    if args.output == '':
        save_path = './'+seq.name+'_prediction.pkl'
    elif args.output[-1] == '/':
        save_path = args.output + seq.name+'_prediction.pkl'
    else:
        save_path = args.output

    print('Buildling input vector...')
    seq.build()
    try:
        seq.seq
        seq.hhm
        seq.dca
    except:
        print('ERROR! unable to properly build input vector. Please check that input is in a3m format')

    total_pred = []
    ctr = 0

    print('Loading ProSPr model(s)...')
    for path in model_paths:
        prospr = ProsprNetwork()
        load_model(prospr, path)
        prospr.to(CUDA)
        print('Model location:',next(prospr.parameters()).device)
        print('Making predictions...')
        pred = predict_domain(data=seq, model=prospr)
        prospr = None

        if total_pred == []:
            total_pred = pred 
        else:
            for key,val in pred.items():
                total_pred[key] += val
        ctr +=1
    
    print('Saving results...')
    avg_pred = dict()
    avg_pred['domain'] = seq.name
    avg_pred['seq'] = seq.seq
    for key,val in total_pred.items():
        avg = val/ctr
        avg_pred[key] = torch.nn.functional.softmax(torch.tensor(avg), dim=0).numpy()
    nets = [p.split('nn/')[-1] for p in model_paths]
    avg_pred['network'] = ', '.join(nets)
    avg_pred['description'] = avg_pred['domain'] + ' predictions made with ' + avg_pred['network'] + ' using default settings: WEIGHTED crop assembly of 10 grids, reported as PROBABILITIES'
    avg_pred['dist_bin_map'] = [0, 4.001, 6.001, 8.001, 10.001, 12.001, 14.001, 16.001, 18.001, 20.001]
    
    save(avg_pred, save_path)
    print('Done! Successfully saved at ', save_path)
    return


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
