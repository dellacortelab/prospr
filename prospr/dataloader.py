import numpy as np
import torch
from prospr.io import load
from prospr.pconf import basedir
from prospr.nn import INPUT_DIM

def seq2mat(s):
    '''Convert a sequence into 21xN matrix. For dataloader'''
    seq_profile = np.zeros((len(s),21),dtype=np.short)
    aa_order = 'ARNDCQEGHILKMFPSTWYVX'
    seq_order= [aa_order.find(letter) for letter in s]
    
    for i,letter in enumerate(seq_order):
        seq_profile[i,letter] = 1
    return seq_profile


def pickled(name = None, i = None, j = None):
    
    pklfilename = basedir +name+ "/" + name +'.pkl' 
    data = load(pklfilename)
    seq = data['seq']
    len_seq = len(seq)

    pssm = data['PSSM']
    hh   = data['HH']
    potts_j = data['J']
    potts_h = data['h']
    potts_score = data['score']
    potts_fn = data['frobenius_norm']
                                                              
    seq_mat = seq2mat(seq) #There is a numpy function to do this
    
    lower_i = max(0, i - 32)
    upper_i = min(len_seq, i + 32)
    irange = upper_i - lower_i
    
    lower_j = max(0, j - 32)
    upper_j = min(len_seq, j + 32)   
    jrange = upper_j - lower_j
    xi = min(32,i)
    yi = min(32,len_seq - i)
        
    xj = min(32,j)
    yj = min(32,len_seq - j)
    
    
    #translate pixels correctly! this is necessary because we map the frame to the zeroed np array!
    vlower_i = lower_i - i + 32
    vupper_i = upper_i - i + 32
    
    vlower_j = lower_j - j + 32
    vupper_j = upper_j - j + 32
    
    #Begin filling input vector
    empty_vector = np.zeros((INPUT_DIM,64,64))
    curr = 0
    
    #first enter potts raw
    potts_j_crop = potts_j[lower_i:upper_i, lower_j:upper_j].reshape((irange,jrange,22*22))
    empty_vector[curr:curr+22*22, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(potts_j_crop,axes=(2,0,1))
    curr+=22*22
    
    #second enter potts score
    empty_vector[curr, vlower_i:vupper_i, vlower_j:vupper_j] = potts_score[lower_i:upper_i, lower_j:upper_j]
    curr+=1
    
    #third enter frobenius norm
    empty_vector[curr, vlower_i:vupper_i, vlower_j:vupper_j] = potts_fn[lower_i:upper_i, lower_j:upper_j]
    curr+=1
       
    #tile i and j
    np.repeat(seq_mat[:,:64,np.newaxis],64,axis=2)
    empty_vector[curr, vlower_i : vupper_i, :] = np.repeat(np.arange(lower_i, upper_i)[:,np.newaxis],64,axis=1)
    curr+=1
    empty_vector[curr, :, vlower_j : vupper_j] = np.repeat(np.arange(lower_j, upper_j)[np.newaxis,:],64,axis=0)
    curr+=1
    
    #potts_h
    empty_vector[curr:curr+22, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(np.repeat(potts_h[lower_i:upper_i,np.newaxis,:], jrange, axis=1),axes=(2,0,1))
    curr += 22
    empty_vector[curr:curr+22, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(np.repeat(potts_h[np.newaxis,lower_j:upper_j,:], irange, axis=0),axes=(2,0,1))
    curr += 22
    
    #fourth enter seq_profile_crop    
    empty_vector[curr:curr+21, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(np.repeat(seq_mat[lower_i:upper_i,np.newaxis,:], jrange, axis=1),axes=(2,0,1))
    curr+=21    
    empty_vector[curr:curr+21, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(np.repeat(seq_mat[np.newaxis,lower_j:upper_j,:], irange, axis=0),axes=(2,0,1))
    curr+=21    
    
    #pssm
    empty_vector[curr:curr+20, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(np.repeat(   pssm[lower_i:upper_i,np.newaxis,:], jrange, axis=1),axes=(2,0,1))
    curr+=20    
    empty_vector[curr:curr+20, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(np.repeat(   pssm[np.newaxis,lower_j:upper_j,:], irange, axis=0),axes=(2,0,1))
    curr+=20
    
    #hh
    empty_vector[curr:curr+30, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(np.repeat(     hh[lower_i:upper_i,np.newaxis,:], jrange, axis=1),axes=(2,0,1))
    curr+=30
    empty_vector[curr:curr+30, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(np.repeat(     hh[np.newaxis,lower_j:upper_j,:], irange, axis=0),axes=(2,0,1))
    curr+=30
    
    #Seq_len
    empty_vector[curr,vlower_i:vupper_i, vlower_j:vupper_j]=len_seq
    curr+=1
    
    input_vector = torch.tensor(empty_vector,dtype=torch.float)
    

    empty_label  = np.zeros((64,64))

    sec_i = np.zeros((64))
    sec_j = np.zeros((64))

    phi_i = np.zeros((64))
    phi_j = np.zeros((64))
 
    psi_i = np.zeros((64))
    psi_j = np.zeros((64))

    bin_dist_label = torch.tensor(empty_label, dtype=torch.long)
    ss_i_label     = torch.tensor(sec_i,       dtype=torch.long)
    ss_j_label     = torch.tensor(sec_j,       dtype=torch.long)
    phi_i_label    = torch.tensor(phi_i,       dtype=torch.long)
    phi_j_label    = torch.tensor(phi_j,       dtype=torch.long)
    psi_i_label    = torch.tensor(psi_i,       dtype=torch.long)
    psi_j_label    = torch.tensor(psi_j,       dtype=torch.long)

    return input_vector, bin_dist_label, ss_i_label, ss_j_label, phi_i_label, phi_j_label, psi_i_label, psi_j_label



def pickled_no_msa(name = None, i = None, j = None):

    pklfilename = basedir +name+ "/" + name +'.pkl' 
    data = load(pklfilename)
    seq = data['seq']
    len_seq = len(seq)
              
    seq_mat = seq2mat(seq) #There is a numpy function to do this
    
    lower_i = max(0, i - 32)
    upper_i = min(len_seq, i + 32)
    irange = upper_i - lower_i
    
    lower_j = max(0, j - 32)
    upper_j = min(len_seq, j + 32)   
    jrange = upper_j - lower_j
    xi = min(32,i)
    yi = min(32,len_seq - i)
        
    xj = min(32,j)
    yj = min(32,len_seq - j)
    
    
    #translate pixels correctly! this is necessary because we map the frame to the zeroed np array!
    vlower_i = lower_i - i + 32
    vupper_i = upper_i - i + 32
    
    vlower_j = lower_j - j + 32
    vupper_j = upper_j - j + 32
    
    #Begin filling input vector
    empty_vector = np.zeros((INPUT_DIM,64,64))
    curr = 0
 
    #tile i and j
    np.repeat(seq_mat[:,:64,np.newaxis],64,axis=2)
    empty_vector[curr, vlower_i : vupper_i, :] = np.repeat(np.arange(lower_i, upper_i)[:,np.newaxis],64,axis=1)
    curr+=1
    empty_vector[curr, :, vlower_j : vupper_j] = np.repeat(np.arange(lower_j, upper_j)[np.newaxis,:],64,axis=0)
    curr+=1
    
    #fourth enter seq_profile_crop    
    empty_vector[curr:curr+21, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(np.repeat(seq_mat[lower_i:upper_i,np.newaxis,:], jrange, axis=1),axes=(2,0,1))
    curr+=21    
    empty_vector[curr:curr+21, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(np.repeat(seq_mat[np.newaxis,lower_j:upper_j,:], irange, axis=0),axes=(2,0,1))
    curr+=21    

    #Seq_len
    empty_vector[curr,vlower_i:vupper_i, vlower_j:vupper_j]=len_seq
    curr+=1
    
    input_vector = torch.tensor(empty_vector,dtype=torch.float)
    
    empty_label  = np.zeros((64,64))
    sec_i = np.zeros((64))
    sec_j = np.zeros((64))

    phi_i = np.zeros((64))
    phi_j = np.zeros((64))
 
    psi_i = np.zeros((64))
    psi_j = np.zeros((64))
 
    bin_dist_label = torch.tensor(empty_label, dtype=torch.long)
    ss_i_label     = torch.tensor(sec_i,       dtype=torch.long)
    ss_j_label     = torch.tensor(sec_j,       dtype=torch.long)
    phi_i_label    = torch.tensor(phi_i,       dtype=torch.long)
    phi_j_label    = torch.tensor(phi_j,       dtype=torch.long)
    psi_i_label    = torch.tensor(psi_i,       dtype=torch.long)
    psi_j_label    = torch.tensor(psi_j,       dtype=torch.long)

    return input_vector, bin_dist_label, ss_i_label, ss_j_label, phi_i_label, phi_j_label, psi_i_label, psi_j_label



def pickled_no_potts(name = None, i = None, j = None):
  
    pklfilename = basedir +name+ "/" + name +'.pkl' 
    data = load(pklfilename)
    seq = data['seq']
    len_seq = len(seq)

    pssm = data['PSSM']
    hh   = data['HH']
                              
    seq_mat = seq2mat(seq) #There is a numpy function to do this
    
    lower_i = max(0, i - 32)
    upper_i = min(len_seq, i + 32)
    irange = upper_i - lower_i
    
    lower_j = max(0, j - 32)
    upper_j = min(len_seq, j + 32)   
    jrange = upper_j - lower_j
    xi = min(32,i)
    yi = min(32,len_seq - i)
        
    xj = min(32,j)
    yj = min(32,len_seq - j)
    
    
    #translate pixels correctly! this is necessary because we map the frame to the zeroed np array!
    vlower_i = lower_i - i + 32
    vupper_i = upper_i - i + 32
    
    vlower_j = lower_j - j + 32
    vupper_j = upper_j - j + 32
    
    #Begin filling input vector
    empty_vector = np.zeros((INPUT_DIM,64,64))
    curr = 0

    #tile i and j
    np.repeat(seq_mat[:,:64,np.newaxis],64,axis=2)
    empty_vector[curr, vlower_i : vupper_i, :] = np.repeat(np.arange(lower_i, upper_i)[:,np.newaxis],64,axis=1)
    curr+=1
    empty_vector[curr, :, vlower_j : vupper_j] = np.repeat(np.arange(lower_j, upper_j)[np.newaxis,:],64,axis=0)
    curr+=1
    
    #fourth enter seq_profile_crop    
    empty_vector[curr:curr+21, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(np.repeat(seq_mat[lower_i:upper_i,np.newaxis,:], jrange, axis=1),axes=(2,0,1))
    curr+=21    
    empty_vector[curr:curr+21, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(np.repeat(seq_mat[np.newaxis,lower_j:upper_j,:], irange, axis=0),axes=(2,0,1))
    curr+=21    
    
    #pssm
    empty_vector[curr:curr+20, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(np.repeat(   pssm[lower_i:upper_i,np.newaxis,:], jrange, axis=1),axes=(2,0,1))
    curr+=20    
    empty_vector[curr:curr+20, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(np.repeat(   pssm[np.newaxis,lower_j:upper_j,:], irange, axis=0),axes=(2,0,1))
    curr+=20
    
    #hh
    empty_vector[curr:curr+30, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(np.repeat(     hh[lower_i:upper_i,np.newaxis,:], jrange, axis=1),axes=(2,0,1))
    curr+=30
    empty_vector[curr:curr+30, vlower_i:vupper_i, vlower_j:vupper_j] = np.transpose(np.repeat(     hh[np.newaxis,lower_j:upper_j,:], irange, axis=0),axes=(2,0,1))
    curr+=30
    
    #Seq_len
    empty_vector[curr,vlower_i:vupper_i, vlower_j:vupper_j]=len_seq
    curr+=1
    
    input_vector = torch.tensor(empty_vector,dtype=torch.float)
    
    empty_label  = np.zeros((64,64))
    sec_i = np.zeros((64))
    sec_j = np.zeros((64))

    phi_i = np.zeros((64))
    phi_j = np.zeros((64))

    psi_i = np.zeros((64))
    psi_j = np.zeros((64))

    bin_dist_label = torch.tensor(empty_label, dtype=torch.long)
    ss_i_label     = torch.tensor(sec_i,       dtype=torch.long)
    ss_j_label     = torch.tensor(sec_j,       dtype=torch.long)
    phi_i_label    = torch.tensor(phi_i,       dtype=torch.long)
    phi_j_label    = torch.tensor(phi_j,       dtype=torch.long)
    psi_i_label    = torch.tensor(psi_i,       dtype=torch.long)
    psi_j_label    = torch.tensor(psi_j,       dtype=torch.long)

    return input_vector, bin_dist_label, ss_i_label, ss_j_label, phi_i_label, phi_j_label, psi_i_label, psi_j_label
