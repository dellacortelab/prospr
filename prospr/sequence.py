import numpy as np
import os
import string
import re
import tensorflow as tf

tf.compat.v1.enable_eager_execution()

def probability(n):
    try:
        counts = int(n)
    except:
        return 0.
    return 2.**(-counts/1000.)

def find_rows(filename):
    with open(filename) as f:
        contents=f.read()
    result = re.search('LENG  (.*) match', contents)
    return int(result.group(1))

def find_hashtag(data):
    for i,line in enumerate(data):
        if line=="#\n":
            return i

def parse_a3m(filename):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    for line in open(filename,"r"):
        if line[0] != '>' and line[0] != '#':
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))

    # convert letters into numbers
    alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
    msa = np.array([list(s) for s in seqs], dtype='|S1').view(np.uint8)
    for i in range(alphabet.shape[0]):
        msa[msa == alphabet[i]] = i

    # treat all unknown characters as gaps
    msa[msa > 20] = 20

    return msa


def fast_dca(msa1hot, weights, penalty = 4.5):

    nr = tf.shape(msa1hot)[0]
    nc = tf.shape(msa1hot)[1]
    ns = tf.shape(msa1hot)[2]

    with tf.name_scope('covariance'):
        x = tf.reshape(msa1hot, (nr, nc * ns))
        num_points = tf.reduce_sum(weights) - tf.sqrt(tf.reduce_mean(weights))
        mean = tf.reduce_sum(x * weights[:,None], axis=0, keepdims=True) / num_points
        x = (x - mean) * tf.sqrt(weights[:,None])
        cov = tf.matmul(tf.transpose(x), x)/num_points

    with tf.name_scope('inv_convariance'):
        cov_reg = cov + tf.eye(nc * ns) * penalty / tf.sqrt(tf.reduce_sum(weights))
        inv_cov = tf.linalg.inv(cov_reg)
        
        x1 = tf.reshape(inv_cov,(nc, ns, nc, ns))
        x2 = tf.transpose(x1, [0,2,1,3])
        features = tf.reshape(x2, (nc, nc, ns * ns))
        
        x3 = tf.sqrt(tf.reduce_sum(tf.square(x1[:,:-1,:,:-1]),(1,3))) * (1-tf.eye(nc))
        apc = tf.reduce_sum(x3,0,keepdims=True) * tf.reduce_sum(x3,1,keepdims=True) / tf.reduce_sum(x3)
        contacts = (x3 - apc) * (1-tf.eye(nc))

    return tf.concat([features, contacts[:,:,None]], axis=2)

def reweight(msa1hot, cutoff):
    """reweight MSA based on cutoff"""
    with tf.name_scope('reweight'):
        id_min = tf.cast(tf.shape(msa1hot)[1], tf.float32) * cutoff
        id_mtx = tf.tensordot(msa1hot, msa1hot, [[1,2], [1,2]])
        id_mask = id_mtx > id_min
        w = 1.0/tf.reduce_sum(tf.cast(id_mask, dtype=tf.float32),-1)
    return w


class Sequence(object):
    def __init__(self, a3m_file, **kwargs):
        self.a3m_file = a3m_file
        self.name = a3m_file.split('.a3m')[0]

    def build(self):
        self.get_seq()
        self.make_hhm()
        self.fast_dca()
        os.system('rm '+self.hhm_file)
    
    def get_seq(self):
        with open(self.a3m_file) as f:
            lns = f.readlines()
            #might not always be the second line in the file
            seq = ''
            l = 0
            while seq == '' and l < len(lns):
                if lns[l][0] == '>':
                    seq = lns[l+1].strip('\n')
                    break
                else:
                    l += 1
            if seq == '':
                print('ERROR! Unable to derive sequence from input a3m file')
                return
            self.seq = seq

    def make_hhm(self):
        #create hhm
        self.hhm_file = 'temp.hhm'
        os.system('hhmake -i '+self.a3m_file+' -o '+self.hhm_file)

        try:
            with open(self.hhm_file) as f:
                data = f.readlines()
        except:
            print('ERROR! Unable to process hhm converted from a3m')
            return

        NUM_COL = 30
        NUM_ROW = find_rows(self.hhm_file)
        pssm = np.zeros((NUM_ROW, NUM_COL))
        line_counter = 0
        start = find_hashtag(data)+5

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
        self.hhm = pssm

    def fast_dca(self):   
        ns           = 21  
        wmin         = 0.8 
        a3m = parse_a3m(self.a3m_file) 
        ncol = a3m.shape[1]
        nrow = tf.Variable(a3m.shape[0])
        msa = tf.Variable(a3m)
        msa1hot  = tf.one_hot(msa, ns, dtype=tf.float32) 
        w = reweight(msa1hot, wmin)                                                                                                     
        f2d_dca = tf.cond(nrow>1, lambda: fast_dca(msa1hot, w), lambda: tf.zeros([ncol,ncol,442], tf.float32))
        f2d_dca = tf.expand_dims(f2d_dca, axis=0).numpy()
        dimensions = f2d_dca.shape
        f2d_dca = f2d_dca.reshape(dimensions[1],dimensions[2],dimensions[3])
        self.dca = f2d_dca.astype('float16')
