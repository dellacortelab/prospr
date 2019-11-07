from prospr.common import findHashTag, probability, findRows
from prospr import pconf
from prospr.nn import *
import numpy as np
import torch
import pickle as pkl
from scipy.io import loadmat

np.set_printoptions(threshold=np.inf)

def save(arr,fileName):
    fileObject = open(fileName, 'wb')
    pkl.dump(arr, fileObject)
    fileObject.close()

def load(fileName):
    fileObject2 = open(fileName, 'rb')
    modelInput = pkl.load(fileObject2)
    fileObject2.close()
    return modelInput

def load_model(fname):
    return torch.load(fname)

class Sequence(object):
    def __init__(s, name, **kwargs):
        """Accepts name as input and expects a corresponding .pssm, hhm, mat, and fastfa txt file (dashes represent gaps), and outputs a pkl file that can be used by ProSPr to predict contacts"""
        s.name = name 
        s.base = pconf.basedir + s.name + "/" + s.name 
        s.data = dict()
        s.outfile = s.base + ".pkl" 
        
    def build(s, args):
        """Builds the pkl file for the protein sequence and saves it.  `args` is an argparse object and it will look, specifically, for a boolean args.potts"""
        s.seq_name()
        s.pssm()
        s.hh()
        s.potts()
        filename = s.base + ".pkl"
        save(s.data, filename)

    def pssm(s): 
        filename = s.base + ".pssm" 
        with open(filename) as f:
            data = f.readlines()
        count = len(data)
        NUM_ROWS = count - 9
        NUM_COL = 20
        matrix = np.zeros((NUM_ROWS,NUM_COL))
        for x in range(NUM_ROWS):
            line = data[x + 3].split()[2:22]
            for i, element in enumerate(line):
                matrix[x,i] = element
        s.data['PSSM'] = matrix

    def hh(s): 
        filename = s.base + ".hhm"
        with open(filename) as f:
            data = f.readlines()
        NUM_COL = 30
        NUM_ROW = findRows(filename)
    
        pssm = np.zeros((NUM_ROW, NUM_COL))
    
        line_counter = 0
    
        start = findHashTag(data)-1
    
        for x in range (0, NUM_ROW * 3):
            if x % 3 == 0:
                line = data[x + start].split()[2:-1]
                for i, element in enumerate(line):
                    prop = probability(element)
                    pssm[line_counter,i] = prop
            elif x % 3 == 1:
                line = data[x+start].split()
                for i, element in enumerate(line):
                    prop = probability(element)
                    pssm[line_counter, i+20] = prop
                line_counter += 1
        s.data['HH'] = pssm
    
         
    def seq_name(s):
        # probably need to pull this filename in from the args
        filename = s.base + ".fasta"
        with open(filename) as f:
            s.data['seq'] = f.readlines()[1]

    def potts(s):
        filename = s.base + ".mat"
        potts = loadmat(filename)
        s.data['J'] = potts['J'].astype('float16')
        s.data['h'] = potts['h']
        s.data['frobenius_norm'] = potts['frobenius_norm']
        s.data['score'] = potts['score']

    
