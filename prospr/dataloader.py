import numpy as np
import torch
import os
import pickle as pkl
import sys
import re
import scipy.io
from Bio import SeqIO
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBParser

from prospr.nn import INPUT_DIM, CROP_SIZE

# Define bin mapping
mapping = np.array([0, 4.001, 6.001, 8.001, 10.001, 12.001, 14.001, 16.001, 18.001, 20.001])

def seq_to_mat(seq):
    """Convert a sequence into Nx21 matrix for input vector"""
    aa_order = 'ARNDCQEGHILKMFPSTWYVX'
    profile = [aa_order.find(letter) for letter in seq]
    return to_categorical(profile, 21)

def to_categorical(x, num_categories):
    """Convert 1d array into 2d array 1-hot encoded"""
    return np.eye(num_categories, dtype='uint8')[x]


def dataloader(data, i, j, train=False, contacts=False): 
    """builds input vector from Sequence object for a specified crop, pads if necessary"""
    seq = data.seq
    len_seq = len(seq)

    lower_i = max(0, i - 32)
    upper_i = min(len_seq, i + 32)
    i_range = upper_i - lower_i
    lower_j = max(0, j - 32)
    upper_j = min(len_seq, j + 32)   
    j_range = upper_j - lower_j

    input_data = np.zeros((INPUT_DIM, i_range, j_range))
    curr = 0

    input_data[curr] = data.dca[lower_i:upper_i, lower_j:upper_j,-1]
    curr+=1

    j_crop = data.dca[lower_i:upper_i, lower_j:upper_j,:-1]
    input_data[curr:curr+21*21] = np.transpose(j_crop,axes=(2,0,1))
    curr+=21*21

    input_data[curr:curr+30] = np.transpose(np.repeat(data.hhm[lower_i:upper_i,np.newaxis,:], j_range, axis=1),axes=(2,0,1))
    curr+=30
    input_data[curr:curr+30] = np.transpose(np.repeat(data.hhm[np.newaxis,lower_j:upper_j,:], i_range, axis=0),axes=(2,0,1))
    curr+=30

    seq_mat = seq_to_mat(seq)
    input_data[curr:curr+21] = np.transpose(np.repeat(seq_mat[lower_i:upper_i,np.newaxis,:], j_range, axis=1),axes=(2,0,1))
    curr+=21    
    input_data[curr:curr+21] = np.transpose(np.repeat(seq_mat[np.newaxis,lower_j:upper_j,:], i_range, axis=0),axes=(2,0,1))
    curr+=21    

    input_data[curr] = np.repeat(np.arange(lower_i, upper_i)[:,np.newaxis],j_range,axis=1)
    curr+=1
    input_data[curr] = np.repeat(np.arange(lower_j, upper_j)[np.newaxis,:],i_range,axis=0)
    curr+=1

    input_data[curr]=len_seq

    vlower_i = lower_i - i + 32
    vupper_i = upper_i - i + 32
    vlower_j = lower_j - j + 32
    vupper_j = upper_j - j + 32
    input_tensor = torch.zeros((INPUT_DIM, CROP_SIZE, CROP_SIZE), dtype=torch.float)
    input_tensor[:,vlower_i:vupper_i,vlower_j:vupper_j] = torch.from_numpy(input_data)

    if train:
        #Create and fill labels
        bin_mat = torch.zeros((CROP_SIZE,CROP_SIZE),dtype=torch.long)
        bin_mat[vlower_i:vupper_i,vlower_j:vupper_j] = torch.from_numpy(data['10_bin_mat'][lower_i:upper_i,lower_j:upper_j].astype('float32'))

        aux_i = dict()
        aux_j = dict()
        aux_i["ss"]  = torch.zeros(CROP_SIZE,dtype=torch.long)
        aux_i["ss"][vlower_i:vupper_i] = torch.from_numpy(data["SS"][lower_i:upper_i])
        aux_j["ss"] = torch.zeros(CROP_SIZE,dtype=torch.long)
        aux_j["ss"][vlower_j:vupper_j] = torch.from_numpy(data["SS"][lower_j:upper_j])
        aux_i["phi"] = torch.zeros(CROP_SIZE,dtype=torch.long)
        aux_i["phi"][vlower_i:vupper_i] = torch.from_numpy(data["phi"][lower_i:upper_i])
        aux_j["phi"] = torch.zeros(CROP_SIZE,dtype=torch.long)
        aux_j["phi"][vlower_j:vupper_j] = torch.from_numpy(data["phi"][lower_j:upper_j])
        aux_i["psi"] = torch.zeros(CROP_SIZE,dtype=torch.long)
        aux_i["psi"][vlower_i:vupper_i] = torch.from_numpy(data["psi"][lower_i:upper_i])
        aux_j["psi"] = torch.zeros(CROP_SIZE,dtype=torch.long)
        aux_j["psi"][vlower_j:vupper_j] = torch.from_numpy(data["psi"][lower_j:upper_j])
        aux_i["asa"]  = torch.zeros(CROP_SIZE,dtype=torch.long)
        aux_i["asa"][vlower_i:vupper_i] = torch.from_numpy(data["ASA"][lower_i:upper_i])
        aux_j["asa"] = torch.zeros(CROP_SIZE,dtype=torch.long)
        aux_j["asa"][vlower_j:vupper_j] = torch.from_numpy(data["ASA"][lower_j:upper_j])
        
        if contacts:
            con_mat = torch.zeros((CROP_SIZE,CROP_SIZE),dtype=torch.long)
            con_mat[vlower_i:vupper_i,vlower_j:vupper_j] = torch.from_numpy(data['contact_mat'][lower_i:upper_i,lower_j:upper_j].astype('float32'))
            return input_tensor, bin_mat, aux_i, aux_j, con_mat
        
        # otherwise
        return input_tensor, bin_mat, aux_i, aux_j

    return input_tensor

#Function definitions
def getDistMat(pdb_file):
    parser = PDBParser(PERMISSIVE=1, QUIET=True)
    structure = parser.get_structure(pdb_file, pdb_file)
    residues = list(structure.get_residues())
    first = residues[0].id[1]
    last = residues[-1].id[1]
    coords = np.empty((last - first + 1,3))
    coords[:] = np.nan
    for residue in residues:
        try:
            if residue.resname == 'GLY':
                coords[residue.id[1]-first] = residue["CA"].get_coord()
            else:
                coords[residue.id[1]-first] = residue["CB"].get_coord()
        except:
            pass
            
    X = coords[None,:,:] - coords[:,None,:]
    X = X**2
    X = np.sum(X,axis=2)
    X = np.sqrt(X)
    return X

def binDistMat(dist_mat): 

    bin_mat = np.empty(dist_mat.shape,dtype=np.int32) 
    L = dist_mat.shape[0] 
    for i in range(L): 
        for j in range(L): 
            if np.isnan(dist_mat[i][j]): 
                bin_mat[i][j] = 0  
            elif dist_mat[i][j] == 0: 
                bin_mat[i][j] = 0 
            else: 
                for pos, bound in enumerate(mapping): 
                    if bound > dist_mat[i][j]: 
                        break 
                    bin_mat[i][j] = pos     
    return bin_mat.astype(np.int8)

def binContacts(dist_mat):
    contact_mat = np.zeros(dist_mat.shape, dtype=np.int8)
    L = dist_mat.shape[0] 
    for i in range(L):
        for j in range(L):
            if not np.isnan(dist_mat[i][j]):
                if dist_mat[i][j] <= 8.0:
                    contact_mat[i][j] = 1
    return contact_mat

'''
dssp tuple:
(dssp index, amino acid, secondary structure, relative ASA, phi, psi,
NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy,
NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy)
'''
def getDSSP(filename):
	parser = PDBParser(PERMISSIVE=1, QUIET=True)
	structure = parser.get_structure(filename,filename)
	dssp = DSSP(structure[0], filename, dssp='mkdssp')  
	return dssp.property_list    

def binDSSP(dssp, seq):
    dssp = list(dssp)

    # Load data without gaps
    ss  = np.zeros(len(seq), dtype=np.uint8)
    asa = np.zeros_like(ss)
    psi = np.zeros_like(ss)
    phi = np.zeros_like(ss)

    ss_symbols = ['H','B','E','G','I','T','S','-']
    ss_mapping = {ss_symbols[i]:i+1 for i in range(8)}

    for i, record in enumerate(dssp):
        try:
            ss[i] = ss_mapping[record[2]]
        except:
            import pdb; pdb.set_trace()
        asa[i] = 1 + int(record[3]*9.999)
        if record[4] != 360.:
            phi[i] = 1 + int((180 + record[4])/10.001)
        if record[5] != 360.:
            psi[i] = 1 + int((180 + record[5])/10.001)
    
    return ss, asa, psi, phi
    
def getPSSM(filename):
    with open(filename) as f:
        data = f.readlines()[3:]
    NUM_ROWS = len(data) - 6
    NUM_COL = 20
    matrix = np.zeros((NUM_ROWS,NUM_COL))
    for i in range(NUM_ROWS):
        matrix[i] = data[i].split()[2:22]
    return matrix

def getSeq(filename):
    with open(filename) as f:
        return f.readlines()[1]

def getHHM(filename):
    with open(filename) as f:
        lines=f.readlines()
    
    for i,line in enumerate(lines):
        if line=="#\n":
            lines = lines[i+5:-2]
            break
            
    NUM_COL = 30
    NUM_ROW = int((len(lines)+1)/3)
            
    profile = np.zeros((NUM_ROW, NUM_COL))          
    for i in range(NUM_ROW):
        row = lines[3*i].split()[2:-1] + lines[3*i+1].split()
        for j in range(NUM_COL):
            if row[j] != '*':
                profile[i,j] = 2.**(-float(row[j])/1000.)
    return profile

def get_label(pdbfile, out_dir='./examples/'):
    id = os.path.basename(os.path.splitext(pdbfile)[0])
    BASE = os.path.join(out_dir, id)
    data = dict()

    PSSM = BASE + '.pssm'
    HHM  = BASE + '.hhm'
    FASTA= BASE + '.fasta'
    MAT  = BASE + '.mat'
    PKL  = BASE + '.pkl'

    # data['seq'] = getSeq(FASTA)
    # data['PSSM'] = getPSSM(PSSM)
    # data['HHM'] = getHHM(HHM)

    # potts = scipy.io.loadmat(MAT)
    # data['J'] = potts['J'].astype(np.float16)
    # data['h'] = potts['h'].astype(np.float16)
    # data['frobenius_norm'] = potts['frobenius_norm'].astype(np.float16)
    # data['corrected_norm'] = potts['corrected_norm'].astype(np.float16)

    with open(pdbfile) as handle:
        sequence = next(SeqIO.parse(handle, "pdb-atom"))
        seq = str(sequence.seq)

    dssp = getDSSP(pdbfile)
    data['DSSP'] = dssp
    ss, asa, psi, phi = binDSSP(dssp, seq)
    data['ss']   = ss
    data['asa']  = asa
    data['phi']  = phi
    data['psi']  = psi

    dist_mat = getDistMat(pdbfile)
    data['dist_mat'] = dist_mat
    data['bin_mat']  = binDistMat(dist_mat)
    data['contact_mat'] = binContacts(dist_mat)
    
    return data
