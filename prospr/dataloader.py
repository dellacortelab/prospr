import numpy as np
import torch

from prospr.nn import INPUT_DIM, CROP_SIZE
    
def seq_to_mat(seq):
    """Convert a sequence into Nx21 matrix for input vector"""
    aa_order = 'ARNDCQEGHILKMFPSTWYVX'
    profile = [aa_order.find(letter) for letter in seq]
    return to_categorical(profile, 21)

def to_categorical(x, num_categories):
    """Convert 1d array into 2d array 1-hot encoded"""
    return np.eye(num_categories, dtype='uint8')[x]


def dataloader(data, i, j): 
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

    return input_tensor
