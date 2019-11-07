import numpy as np
from .dataloader import pickled, pickled_no_msa, pickled_no_potts
from prospr.common import norm, make_uint8
from prospr.io import load
from prospr.nn import *
from prospr.pconf import basedir

def domain(name, model, stride = 1, network_type = 1):
    
    BATCH_SIZE = 1
    SS_BINS = 9
    ANGLE_BINS = 37
    DIST_BINS = 64
    pklfilename = basedir +name+ "/" + name +'.pkl' 
    data = load(pklfilename)
    
    seq = data['seq']
    seq_len = len(seq)
    
    
    ss_sum = np.zeros([SS_BINS,seq_len])
    phi_sum = np.zeros([ANGLE_BINS,seq_len])
    psi_sum = np.zeros([ANGLE_BINS,seq_len])
    dist_sum = np.zeros([DIST_BINS,seq_len,seq_len])
    
    dim2_ct = np.zeros([ANGLE_BINS,seq_len])
    dim3_ct = np.zeros([DIST_BINS,seq_len,seq_len])
    
    criterion = nn.CrossEntropyLoss()
    
    crops_per_len = seq_len // stride
    if seq_len % stride > 0:
        crops_per_len += 1
    total_crops = crops_per_len * crops_per_len
    print('\nMaking predictions for', name, 'using network', network_type, 'with stride', stride)
    print('Sequence length is', seq_len)
    print('Processing', str(total_crops), 'total crops...')
    print('\t please note this will take longer because not using a(ny) GPU(s)!')

    crop_num = 1    
    last_update = 0
    
    i = j = 0
    while i < seq_len:
        while j < seq_len:

            profile = torch.zeros([BATCH_SIZE,INPUT_DIM,CROP_SIZE,CROP_SIZE], dtype = torch.float)
            labels = torch.zeros([BATCH_SIZE,CROP_SIZE,CROP_SIZE], dtype=torch.long)
            
            for b in range(BATCH_SIZE):
                if network_type == 1: #ProSPr
                    input_vector, label, label_ss_i, label_ss_j, label_phi_i, label_phi_j, label_psi_i, label_psi_j = pickled(name, i, j)
                elif network_type == 2: #ProSPr2
                    input_vector, label, label_ss_i, label_ss_j, label_phi_i, label_phi_j, label_psi_i, label_psi_j = pickled_no_potts(name, i, j)
                elif network_type == 3: #ProSPr3
                    input_vector, label, label_ss_i, label_ss_j, label_phi_i, label_phi_j, label_psi_i, label_psi_j = pickled_no_msa(name, i, j)
                else:
                    print(network_type, type(network_type))
                profile[b,:] = input_vector
                labels[b,:] = torch.tensor(label.detach().numpy(), dtype=torch.long)

            outputs, output_ss_i, output_ss_j, output_phi_i, output_phi_j, output_psi_i, output_psi_j = model(profile)
            predictions = outputs.reshape((BATCH_SIZE,DIST_BINS, 64,64))

            ss_i = output_ss_i.detach().numpy()[0].reshape(SS_BINS,CROP_SIZE)
            ss_j = output_ss_j.detach().numpy()[0].reshape(SS_BINS,CROP_SIZE)
            phi_i = output_phi_i.detach().numpy()[0].reshape(ANGLE_BINS,CROP_SIZE)
            phi_j = output_phi_j.detach().numpy()[0].reshape(ANGLE_BINS,CROP_SIZE)
            psi_i = output_psi_i.detach().numpy()[0].reshape(ANGLE_BINS,CROP_SIZE)
            psi_j = output_psi_j.detach().numpy()[0].reshape(ANGLE_BINS,CROP_SIZE)
            dist = predictions.detach().numpy()[0]
            
            if i > (seq_len-32): #this needs to happen first because depends upon total size!!
                #right side needs padding cropped off
                ss_i = ss_i[:,:(CROP_SIZE-(i-(seq_len-32)))]
                phi_i = phi_i[:,:(CROP_SIZE-(i-(seq_len-32)))]
                psi_i = psi_i[:,:(CROP_SIZE-(i-(seq_len-32)))]
                dist = dist[:,:(CROP_SIZE-(i-(seq_len-32))),:]
            if i < 32:
                #left side needs padding cropped off
                ss_i = ss_i[:,(32-i):]
                phi_i = phi_i[:,(32-i):]
                psi_i = psi_i[:,(32-i):]
                dist = dist[:,(32-i):,:]
            
            if j > (seq_len-32): #this needs to happen first because depends upon total size!!
                #right side needs padding cropped off
                ss_j = ss_j[:,:(CROP_SIZE-(j-(seq_len-32)))]
                phi_j = phi_j[:,:(CROP_SIZE-(j-(seq_len-32)))]
                psi_j = psi_j[:,:(CROP_SIZE-(j-(seq_len-32)))]
                dist = dist[:,:,:(CROP_SIZE-(j-(seq_len-32)))]
            if j < 32:
                #left side needs padding cropped off
                ss_j = ss_j[:,(32-j):]
                phi_j = phi_j[:,(32-j):]
                psi_j = psi_j[:,(32-j):]
                dist = dist[:,:,(32-j):]
                
            ss_i = norm(ss_i)
            ss_j = norm(ss_j)
            phi_i = norm(phi_i)
            phi_j = norm(phi_j)
            psi_i = norm(psi_i)
            psi_j = norm(psi_j)
            dist = norm(dist)
            
            dim_i = dist.shape[1]
            dim_j = dist.shape[2]
            start_i = np.max([0, i-32])
            start_j = np.max([0, j-32])
            
            ss_sum[:,start_i:(start_i+int(dim_i))] += ss_i
            ss_sum[:,start_j:(start_j+int(dim_j))] += ss_j
            phi_sum[:,start_i:(start_i+int(dim_i))] += phi_i
            phi_sum[:,start_j:(start_j+int(dim_j))] += phi_j
            psi_sum[:,start_i:(start_i+int(dim_i))] += psi_i
            psi_sum[:,start_j:(start_j+int(dim_j))] += psi_j
            dist_sum[:,start_i:(start_i+int(dim_i)),start_j:(start_j+int(dim_j))] += dist
            
            dim2_ct[:,start_i:(start_i+int(dim_i))] += np.ones(phi_i.shape)
            dim2_ct[:,start_j:(start_j+int(dim_j))] += np.ones(phi_j.shape)
            dim3_ct[:,start_i:(start_i+int(dim_i)),start_j:(start_j+int(dim_j))] += np.ones(dist.shape)
            print('.', end='')
            progress = (crop_num - last_update) / total_crops
            if progress > .1:
                last_update = crop_num
                print('   '+str((crop_num/total_crops)*100//1)+'%'+' complete')
            crop_num += 1
            j += stride
        j = 0
        i += stride

    dist_ct = dim3_ct
    ss_ct = dim2_ct[:9,:]
    angle_ct = dim2_ct
    
    dist_avg = dist_sum / dist_ct
    ss_avg = ss_sum / ss_ct
    phi_avg = phi_sum / angle_ct
    psi_avg = psi_sum / angle_ct

    dist_label = np.zeros((seq_len, seq_len))
    ss_label = np.zeros((seq_len))
    phi_label = np.zeros((seq_len))
    psi_label = np.zeros((seq_len))
    
    
    dist_pred = torch.tensor(dist_avg.reshape([1,DIST_BINS,seq_len,seq_len]), dtype=torch.float)
    dist_label = torch.tensor(make_uint8(dist_label).reshape([1,seq_len,seq_len]), dtype=torch.long)
    ss_pred = torch.tensor(ss_avg.reshape([1,SS_BINS,seq_len]), dtype=torch.float)
    ss_label = torch.tensor(ss_label.reshape([1,seq_len]), dtype=torch.long)
    phi_pred = torch.tensor(phi_avg.reshape([1,ANGLE_BINS,seq_len]), dtype=torch.float)
    phi_label = torch.tensor(phi_label.reshape([1,seq_len]), dtype=torch.long)
    psi_pred = torch.tensor(psi_avg.reshape([1,ANGLE_BINS,seq_len]), dtype=torch.float)
    psi_label = torch.tensor(psi_label.reshape([1,seq_len]), dtype=torch.long)
    
    loss_dist = criterion(dist_pred, dist_label).detach().numpy()
    
    dist_prob = torch.nn.functional.softmax(dist_pred, dim=1)
    dist_prob = dist_prob.reshape(DIST_BINS, seq_len, seq_len)    
    return dist_prob, loss_dist
