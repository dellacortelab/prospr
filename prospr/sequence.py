import numpy as np
import os
import string
import re
import tensorflow as tf
from Bio import SeqIO
from Bio.PDB.DSSP import DSSP
from Bio.PDB import PDBParser

tf.compat.v1.enable_eager_execution()
mapping = np.array([0, 4.001, 6.001, 8.001, 10.001, 12.001, 14.001, 16.001, 18.001, 20.001])

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
        ss[i] = ss_mapping[record[2]]

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


class Sequence(object):
    def __init__(self, a3m_file, include_labels=False, subsample_hmm_percent=1.0, **kwargs):
        self.a3m_file = a3m_file
        self.name = os.path.basename(a3m_file.split('.a3m')[0])
        self.subsample_hmm_percent = subsample_hmm_percent
        self.include_labels = include_labels

    def build(self, pdbfile=None):
        self.subsample_a3m()
        self.get_seq()
        self.make_hhm()
        self.fast_dca()
        os.system('rm '+self.hhm_file)
        if self.include_labels:
            self.get_label(pdbfile)

    def subsample_a3m(self):
        if self.subsample_hmm_percent < 1.0:
            subsample_a3m = os.path.join(os.path.dirname(self.a3m_file), self.name + '_subsample.a3m')
            with open(self.a3m_file) as f:
                lns = f.readlines()
                n_msas = (len(lns) // 2) - 1 # 1st line is the sequence itself
                selected_indices = np.random.choice(n_msas, size=int(n_msas*self.subsample_hmm_percent), replace=False)
                with open(subsample_a3m, 'w') as out_file:
                    # Write the sequence
                    out_file.write(''.join(lns[:2]))
                    for i in selected_indices:
                        out_file.write(''.join(lns[2*i:2*i+2]))
                self.a3m_file = subsample_a3m

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
        os.system('hhmake -i '+self.a3m_file+' -o '+self.hhm_file+' -v 0')

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

    def get_label(self, pdbfile, out_dir='./examples/'):
        id = os.path.basename(os.path.splitext(pdbfile)[0])
        # BASE = os.path.join(out_dir, id)
        data = dict()

        # PSSM = BASE + '.pssm'
        # HHM  = BASE + '.hhm'
        # FASTA= BASE + '.fasta'
        # MAT  = BASE + '.mat'
        # PKL  = BASE + '.pkl'

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
        
        self.label_data = data
