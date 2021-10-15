import torch
import torch.nn as nn
import numpy as np
import random
import torch.optim as optim
import os
import torch.utils.data
import time

from prospr.io import save, load
from prospr.nn import ProsprNetwork
from prospr.dataloader import get_tensors
from prospr.sequence import Sequence

import warnings
from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)
warnings.simplefilter('ignore', FutureWarning)
# ** HELPER FUNCTIONS ** #

def check_param(model):
    a = list(model.parameters())
    s = 0
    for i in a[:]:
        s += np.product(i.shape)
    print('Model has', (s/1000000),  'million parameters')

def save_model(model, fname):
    torch.save(model.state_dict(),fname)

def load_model(model, fname, device):
    model.load_state_dict(torch.load(fname,map_location=device))
    return model

def prep_domain(domain, mods, crop_size, data_path):
    # for epoch enumeration, load name2seq for whole training set
    name2seq = load(os.path.join(data_path, 'training-name2seq-small.pkl'))#'TRAIN_name2seq.pkl')
    # name2seq = load(os.path.join(data_path, 'training-name2seq-EXPANDED_21may.pkl'))#'TRAIN_name2seq.pkl')
    
    seq = name2seq[domain]
    seq_len = len(seq)

    crop_list = []

    #calculate num crops along one dimension
    num_crops = seq_len // crop_size
    remainder = seq_len % crop_size
    start_i = 0
    start_j = 0
    if remainder > 0:
        num_crops += 1

    index_options = mods[seq_len%crop_size]

    start_i = random.choice(index_options)
    start_j = random.choice(index_options)
    
    i = start_i
    j = start_j
    while i < seq_len:
        while j < seq_len:
            crop_list.append((domain, i, j))
            j += 64
        j = start_j
        i += 64

    try:
        assert(len(crop_list) == num_crops**2)
    except:
        print(len(crop_list),num_crops**2)
    return crop_list

def make_epoch_stack(training_list, model_name, mods, crop_size, epoch, log_path, data_path):
    big_list = []
    for domain in training_list:
        try:
            small_list = prep_domain(domain, mods, crop_size, data_path)
            big_list = big_list + small_list
        except Exception as e:
            print(e)
            print(domain, 'could not be added to epoch crop list!')
            with open(os.path.join(log_path, model_name + 'error_log.txt'), 'a+') as f:
                f.write(str(epoch) + '\t'+domain+'\t'+'could not be added to epoch crop list!\n')
    print('Epoch',epoch,'list has', len(big_list), 'total crops!')
    return big_list

def build_batch_string(crops):
    desc = ''
    for crop in crops:
        desc += (crop[0]+ '_'+str(crop[1])+'_'+str(crop[2])+'\t')
    return desc

def train(args):

    # ** HYPERPARAMETERS ** #

    # GLOBAL VARIBALES
    INPUT_DIM = 547
    DIST_BINS = 10 
    AUX_BINS = 94
    DROPOUT_RATE = 0.15 #keeps 85%

    LEARNING_RATE = 0.001

    IDEAL_BATCH_SIZE = 6
    NUM_EPOCHS = 5

    dist_loss_factor = 15
    ss_loss_factor = 0.5
    torsion_loss_factor = 0.25
    asa_loss_factor = 0.5

    WAIT_TIME = 200 #seconds


    PADDING = args.crop_size // 2

    #attempting to make a rule
    mods = [] #will contain list of possible start indices indexed by L % args.crop_size
    #therefore should have args.crop_size total elements at the end

    #do first, edge case:
    mods.append([PADDING])

    #initialize entire index vector
    indices = [i for i in range(0,PADDING+1)]

    #now do 1->PADDING
    for i in range(1,PADDING+1):
        mods.append(indices[0:i])

    #now do PADDING->end
    for i in range(PADDING+1, args.crop_size):
        mods.append(indices[i-PADDING:])

    #run length checks
    for i, thing in enumerate(mods):
        if i == 0:
            assert(len(thing) == 1)
        elif i <= PADDING:
            assert(len(thing) == i)
        else:
            assert(len(thing) + i -1 == args.crop_size)
    print('Length check on start indices is good!')

    data_path = os.path.join(args.base_data_path, 'training')
    model_path = './nn'
    log_path = os.path.join(args.base_data_path, 'logs')

    # training_list = load(os.path.join(data_path, 'training-set-EXPANDED_21may.pkl'))#'good_TRAIN_domains_apr29.pkl')
    training_list = load(os.path.join(data_path, 'training-set-small.pkl'))

    p = ProsprNetwork()

    #start_model = 
    #p = load_model(p, start_model, args.device)

    p = p.to(args.device)
    p = p.train()

    CRITERION = nn.CrossEntropyLoss()
    OPTIMIZER = optim.Adam(p.parameters(), lr=LEARNING_RATE )

    check_param(p)

    domain_path = data_path + 'current_epoch/SUB_'

    for epoch in range(NUM_EPOCHS):
        print('Starting epoch',epoch)
        if epoch == 5:
            print('changing LR to 0.0005')
            LEARNING_RATE  = 0.0005
            OPTIMIZER = optim.Adam(p.parameters(), lr=LEARNING_RATE )
        elif epoch == 15:
            print('changing LR to 0.0001')
            LEARNING_RATE  = 0.0001
            OPTIMIZER = optim.Adam(p.parameters(), lr=LEARNING_RATE )

        iteration = 0

        BATCH_SIZE = IDEAL_BATCH_SIZE

        random.shuffle(training_list)
        crop_list = make_epoch_stack(training_list, model_name=args.model_name, mods=mods, crop_size=args.crop_size, epoch=epoch, log_path=log_path, data_path=data_path) #this contains ALL the crops for the WHOLE epoch!
        done = False 
    
        while not done:
            if len(crop_list) == 0:
                done = True
                break
            if len(crop_list) < BATCH_SIZE:
                BATCH_SIZE = len(crop_list)
                done = True #because shouldn't repeat loop again afterwards (for this epoch)
        
            take_step = False
            try:
                input_vector = torch.zeros([BATCH_SIZE,INPUT_DIM,args.crop_size,args.crop_size], dtype = torch.float, device=args.device)
                labels_dist = torch.zeros([BATCH_SIZE,args.crop_size,args.crop_size], dtype=torch.long, device = args.device)
                labels_ss_i = torch.zeros([BATCH_SIZE,args.crop_size], dtype=torch.long, device = args.device)
                labels_ss_j = torch.zeros([BATCH_SIZE,args.crop_size], dtype=torch.long, device = args.device)
                labels_phi_i = torch.zeros([BATCH_SIZE,args.crop_size], dtype=torch.long, device = args.device)
                labels_phi_j = torch.zeros([BATCH_SIZE,args.crop_size], dtype=torch.long, device = args.device)
                labels_psi_i = torch.zeros([BATCH_SIZE,args.crop_size], dtype=torch.long, device = args.device)
                labels_psi_j = torch.zeros([BATCH_SIZE,args.crop_size], dtype=torch.long, device = args.device)
                labels_asa_i = torch.zeros([BATCH_SIZE,args.crop_size], dtype=torch.long, device = args.device)
                labels_asa_j = torch.zeros([BATCH_SIZE,args.crop_size], dtype=torch.long, device = args.device)

                batch_crop_info = []

                for batch in range(BATCH_SIZE):
                    crop = crop_list.pop(0)
                    domain, i, j = crop
                    
                    sequence = Sequence(os.path.join(data_path, 'a3ms', domain + '.a3m'), subsample_hmm_percent=.5, include_labels=True)
                    sequence.build(pdbfile=os.path.join(data_path, 'pdbs', domain + '.pdb'))
                    iv, dist_label, aux_i_labels, aux_j_labels = get_tensors(sequence, i, j, train=True)

                    batch_crop_info.append((domain,i,j))
                    input_vector[batch,:] = iv
                    labels_dist[batch,:] = dist_label

                    labels_ss_i[batch,:] = aux_i_labels['ss']
                    labels_phi_i[batch,:] = aux_i_labels['phi']
                    labels_psi_i[batch,:] = aux_i_labels['psi']
                    labels_asa_i[batch,:] = aux_i_labels['asa']

                    labels_ss_j[batch,:] = aux_j_labels['ss']
                    labels_phi_j[batch,:] = aux_j_labels['phi']
                    labels_psi_j[batch,:] = aux_j_labels['psi']
                    labels_asa_j[batch,:] = aux_j_labels['asa']
                    
                take_step = True

            except Exception as e:
                print(e)
                #write to error log
                batch_desc = build_batch_string(batch_crop_info)
                print('BATCH ASSEMBLY PROBLEM', batch_desc)
                with open(os.path.join(log_path, args.model_name + 'error_log.txt'), 'a+') as f:
                    f.write('BATCH ASSEMBLY PROBLEM' + '\t' +str(epoch) + '\t'+str(iteration)+'\t'+str(BATCH_SIZE)+'\t'+batch_desc+'\t'+domain+'\t'+str(i)+'\t'+str(j)+'\n')
                #if ran into a problem, (shouldn't happen, but...) it will keep popping the next crop off util continues
                
            if take_step:
                try:
                    OPTIMIZER.zero_grad()
                    pred_dist, pred_aux_i, pred_aux_j = p(input_vector)

                    # calculate the loss and optimize
                    dist_loss = CRITERION(pred_dist, labels_dist)
                    ss_loss = CRITERION(pred_aux_i['ss'], labels_ss_i) + CRITERION(pred_aux_j['ss'], labels_ss_j)
                    phi_loss = CRITERION(pred_aux_i['phi'], labels_phi_i) + CRITERION(pred_aux_j['phi'], labels_phi_j)
                    psi_loss = CRITERION(pred_aux_i['psi'], labels_psi_i) + CRITERION(pred_aux_j['psi'], labels_psi_j)
                    asa_loss = CRITERION(pred_aux_i['asa'], labels_asa_i) + CRITERION(pred_aux_j['asa'], labels_asa_j)

                    loss =  (dist_loss_factor * dist_loss    
                                +  ss_loss_factor * ss_loss 
                                +  torsion_loss_factor * (phi_loss + psi_loss)
                                +  asa_loss_factor * asa_loss)

                    loss.backward()
                    OPTIMIZER.step()

                    #format loss info for printing and saving
                    info = (str(epoch) + '\t' + 
                            str(iteration) + '\t' +
                            "{:.5f}\t".format(loss.item()) +
                            str(dist_loss_factor) + '\t' +
                            "{:.5f}\t".format(dist_loss.item()) +
                            str(ss_loss_factor) + '\t' +
                            "{:.5f}\t".format(ss_loss.item()) +
                            str(torsion_loss_factor) + '\t' +
                            "{:.5f}\t".format(phi_loss.item()) +
                            "{:.5f}\t".format(psi_loss.item()) +
                            str(asa_loss_factor) + '\t' +
                            "{:.5f}\t".format(asa_loss.item()) +
                            str(LEARNING_RATE) )

                    print(info)

                    #save losses in training log
                    with open(os.path.join(log_path, args.model_name + 'loss_log.txt'), 'a+') as f:
                        f.write(info+'\n')

                    #record batch crop information
                    batch_desc = build_batch_string(batch_crop_info)
                    with open(os.path.join(log_path, args.model_name + 'crop_log.txt'), 'a+') as f:
                        f.write(str(epoch) + '\t'+str(iteration)+'\t'+str(BATCH_SIZE)+'\t'+batch_desc+'\n')

                except:
                    print('Error somewhere in taking optimization step!')
                    batch_desc = build_batch_string(batch_crop_info)
                    with open(os.path.join(log_path, args.model_name + 'error_log.txt'), 'a+') as f:
                        f.write('OPTIMIZATION STEP PROBLEM'+'\t'+str(epoch) + '\t'+str(iteration)+'\t'+str(BATCH_SIZE)+'\t'+batch_desc+'\t'+domain+'\t'+str(i)+'\t'+str(j)+'\n')

                iteration += 1

        save_model(p, os.path.join(model_path, args.model_name+'EPOCH-END_'+str(epoch)+'.pt'))
        
        if args.multi_model:
            #now do the new stuff to play nicely with training data generation also on ribo, automatically
            # first make the ready file for this network
            os.makedirs('./go_ahead', exist_ok=True)
            with open('./go_ahead/1.4-X_GO.txt', 'a+') as f:
                f.write(' test ')
            # now wait around until all three networks are done, and monitor_epochs.sh moves things around and deletes the go_ahead/ files
            do_next_epoch = False
            while not do_next_epoch:
                n = len(os.listdir('./go_ahead/'))
                if n > 0:
                    if n == 3:
                        print('All three networks ready, waiting on training data...')
                    else:
                        print('Only ',n,'/3 networks ready for next epoch. Waiting....')
                    time.sleep(WAIT_TIME)
                elif n == 0:
                    do_next_epoch = True
                else:
                    print('BIG PROBLEMS somehow have negative # of files in go_ahead/ ?????')
            print('Everything ready for next epoch! Continuing on...')
