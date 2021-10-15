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

def seq_to_mat(seq):
    """Convert a sequence into Nx21 matrix for input vector"""
    aa_order = 'ARNDCQEGHILKMFPSTWYVX'
    profile = [aa_order.find(letter) for letter in seq]
    return to_categorical(profile, 21)

def to_categorical(x, num_categories):
    """Convert 1d array into 2d array 1-hot encoded"""
    return np.eye(num_categories, dtype='uint8')[x]


def get_tensors(sequence, i, j, train=False, contacts=False): 
    """builds input vector from Sequence object for a specified crop, pads if necessary"""
    seq = sequence.seq
    len_seq = len(seq)

    lower_i = max(0, i - 32)
    upper_i = min(len_seq, i + 32)
    i_range = upper_i - lower_i
    lower_j = max(0, j - 32)
    upper_j = min(len_seq, j + 32)   
    j_range = upper_j - lower_j

    input_data = np.zeros((INPUT_DIM, i_range, j_range))
    curr = 0

    input_data[curr] = sequence.dca[lower_i:upper_i, lower_j:upper_j,-1]
    curr+=1

    j_crop = sequence.dca[lower_i:upper_i, lower_j:upper_j,:-1]
    input_data[curr:curr+21*21] = np.transpose(j_crop,axes=(2,0,1))
    curr+=21*21

    input_data[curr:curr+30] = np.transpose(np.repeat(sequence.hhm[lower_i:upper_i,np.newaxis,:], j_range, axis=1),axes=(2,0,1))
    curr+=30
    input_data[curr:curr+30] = np.transpose(np.repeat(sequence.hhm[np.newaxis,lower_j:upper_j,:], i_range, axis=0),axes=(2,0,1))
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
        data = sequence.label_data
        #Create and fill labels
        dist_label = torch.zeros((CROP_SIZE,CROP_SIZE),dtype=torch.long)
        dist_label[vlower_i:vupper_i,vlower_j:vupper_j] = torch.from_numpy(data['bin_mat'][lower_i:upper_i,lower_j:upper_j].astype('float32'))

        aux_i = dict()
        aux_j = dict()
        aux_i["ss"]  = torch.zeros(CROP_SIZE,dtype=torch.long)
        aux_i["ss"][vlower_i:vupper_i] = torch.from_numpy(data["ss"][lower_i:upper_i])
        aux_j["ss"] = torch.zeros(CROP_SIZE,dtype=torch.long)
        aux_j["ss"][vlower_j:vupper_j] = torch.from_numpy(data["ss"][lower_j:upper_j])
        aux_i["phi"] = torch.zeros(CROP_SIZE,dtype=torch.long)
        aux_i["phi"][vlower_i:vupper_i] = torch.from_numpy(data["phi"][lower_i:upper_i])
        aux_j["phi"] = torch.zeros(CROP_SIZE,dtype=torch.long)
        aux_j["phi"][vlower_j:vupper_j] = torch.from_numpy(data["phi"][lower_j:upper_j])
        aux_i["psi"] = torch.zeros(CROP_SIZE,dtype=torch.long)
        aux_i["psi"][vlower_i:vupper_i] = torch.from_numpy(data["psi"][lower_i:upper_i])
        aux_j["psi"] = torch.zeros(CROP_SIZE,dtype=torch.long)
        aux_j["psi"][vlower_j:vupper_j] = torch.from_numpy(data["psi"][lower_j:upper_j])
        aux_i["asa"]  = torch.zeros(CROP_SIZE,dtype=torch.long)
        aux_i["asa"][vlower_i:vupper_i] = torch.from_numpy(data["asa"][lower_i:upper_i])
        aux_j["asa"] = torch.zeros(CROP_SIZE,dtype=torch.long)
        aux_j["asa"][vlower_j:vupper_j] = torch.from_numpy(data["asa"][lower_j:upper_j])
        
        if contacts:
            con_mat = torch.zeros((CROP_SIZE,CROP_SIZE),dtype=torch.long)
            con_mat[vlower_i:vupper_i,vlower_j:vupper_j] = torch.from_numpy(data['contact_mat'][lower_i:upper_i,lower_j:upper_j].astype('float32'))
            return input_tensor, dist_label, aux_i, aux_j, con_mat
        
        # otherwise
        return input_tensor, dist_label, aux_i, aux_j

    return input_tensor

