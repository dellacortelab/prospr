import numpy as np
import re

def probability(n):
    try:
        counts = int(n)
    except:
        return 0.
    return 2.**(-counts/1000.)

def findRows(filename):
    with open(filename) as f:
        contents=f.read()
    result = re.search('LENG  (.*) match', contents)
    return int(result.group(1))

def findHashTag(data):
    for i,line in enumerate(data):
        if line=="#\n":
            return i

def check_param(model):
    a = list(model.parameters())
    s = 0
    for i in a[:]:
        s += np.product(i.shape)
    print('Model has', (s/1000000),  'million parameters')

def ohe_aa(aa):
    #returns a list of 20 elements, where one is 1, rest 0
    order = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V','X']
    
    aa2int = dict()
    for i,aa in enumerate(order):
        aa2int[aa] = i
    ohe = [ 0 for i in range(21)]
    #TODO: rethink gap treatment **
    if aa == '-':
        aa = 'X'
    ohe[aa2int[aa]] = 32
    return ohe
    
def tripple_to_letter(tripple):
#I want to encode amino acids through integer number 0 - 19, here is the dict for it, 20 is for 'X' or unnatural AA
    letter_code = {'SER' : 'S', 'ARG' : 'R', 'HIS' : 'H', 'LYS' : 'K', 'ASP' : 'D',
                   'GLU' : 'E', 'THR' : 'T', 'ASN' :'N', 'GLN' : 'Q', 'CYS' : 'C',
                   'SEC' : 'U', 'GLY' : 'G', 'PRO' : 'P', 'ALA' : 'A', 'VAL' : 'V', 
                   'ILE' : 'I', 'LEU' : 'L', 'MET' : 'M', 'PHE' : 'F', 'TYR' : 'Y',
                   'TRP' : 'W', 'UNK' : 'X'}
    tripple_code = {v:k for k, v in letter_code.items()}
    try:
        return letter_code[tripple]
    except:
        return 'X'


def make_uint8(myobj):
    #helper fct to convert from int8 to unit8
    return myobj.astype('uint8')   


def norm(thing):
    mean = np.mean(thing)
    stdev = np.std(thing)
    return (thing - mean) / stdev
