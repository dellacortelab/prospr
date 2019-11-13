import argparse
import os
import subprocess
import multiprocessing
import torch
from datetime import datetime
from argparse import RawTextHelpFormatter
import warnings
warnings.simplefilter("ignore")
import pyfiglet
from prospr.io import load_model, Sequence, save
from prospr.nn import *
from prospr.prediction import domain
from prospr.psiblast import PsiblastManager
from prospr.hhblits import *

allowed = '-ACDEFGHIKLMNPQRSTVWY'

ascii_banner = pyfiglet.figlet_format("Welcome to\n           ProSPr!")
desc_usg = ascii_banner + '''
WARNING: Reviewer edition, limited functionality

usage: docker run -t prospr/prospr <command> <domain>

To get data out, and use your own training data, you need to include a folder.
Here's an example of how to do that:
docker run -t -v /path/to/local/data:/data prospr/prospr <domain> <network> <stride>


To use prospr on your own sequences:
docker run -t -v /path/to/local/data:/data prospr/prospr -t --a3mfile /data/my_file.a3m
and place a PSIBLAST database under /data/psiblast and a uniclust database under /data/hhblits.  ProSPr will download the database if it is missing (250GB and 25GB respectively).
'''

parser = argparse.ArgumentParser(description=desc_usg, usage=argparse.SUPPRESS, formatter_class=RawTextHelpFormatter)
subparsers = parser.add_subparsers(help="""Available commands are:
 build <domain> - builds a sequence from a fasta file that ProSPr can use for predictions 
 run <domain> <stride>  - uses a pkl file for a domain to predict contacts""", dest='command')
run_parser = subparsers.add_parser('run')
run_parser.add_argument('domain', help='CASP13 target domain id (to see list of available samples, run with any invalid char or string')
run_parser.add_argument('-n','--network', help='Which neural network do you want to use to make the predictions.  ProSPr provides the following:  full, no-potts, and no-msa.  full is default', default='full')
run_parser.add_argument('-s','--stride', help='stride over which crops of domain are predicted and averaged, integer 1-30.\nWARNING: Using a small stride may result in very long processing time! Suggested for quick prediction: 25', type=int, default=25)
train_parser = subparsers.add_parser('build')
train_parser.add_argument('domain', help='Looks for /data/<domain>/<domain>.fasta for a sequence and produces a pkl file ProSPr can use.  Please note, gaps in the sequence are denoted by a -, not an X')
train_parser.add_argument('-j', '--threads', help='Number of threads for HHBlits/PSI-BLAST runs', default=1, type=int)

 
def main(args):
    casp_domains = os.listdir(pconf.basedir + args.domain)
    if (args.domain+'.pkl') not in casp_domains:
        print('ERROR! Invalid target, please provide a fasta file and build the sequence' )
        print('Options:')
        for thing in casp_domains:
            if '.pkl' in thing:
                print('\t',thing[:-4])
        return
    if (args.stride < 1) or (args.stride > 30):
        print('ERROR! Invalid stride')
        print('Options:\n\tinteger (0,30]')
        return
    network = load_model('/opt/nn/ProSPr_' + args.network + ".nn")
    dist_pred, dist_loss = domain(args.domain, network, args.stride)
    p = dist_pred.detach().numpy()
    save_path = pconf.basedir+args.domain +"/"
    save(p, save_path + args.domain +'_'+'_'+str(args.stride)+'_predictions.pkl')
    print("contact predictions saved in %s" % save_path)
    print('\nAnalyze prediction results: https://github.com/dellacortelab/prospr\n')
    return
 
def build(args):
    p = PsiblastManager()
    if hhdb_dl_present():
        if hhdb_db_present():
            pass
        else:
            print("uniclust database appears to be downloaded, extracting. . .")
            #hhdb_unzip(pconf.basedir+"hhblits/uniclust30_2018_08_hhsuite.tar.gz")
    else:
        while True:
            res = input("Would you like to download uniclust for hhblits? (25GB) [y/n]: ")
            if res == "y":
                hhdb_install()
                break
            elif res == "n":
                print("please install a database for hhblits in your " + pconf.basedir +"hhblits directory")
                return

    b = BlitsAndPottsRunner(args.domain, n_threads=args.threads)
    print("[%s] Running hhblits. . ." % datetime.now())
    b.start() 
    if p.simple_check_db() and p.decompressed():
        print("Psiblast database present and ready to use")
        print("[%s] Building pssm file" % datetime.now())
        p.build_query(args.domain, n_threads=args.threads)
        print("[%s] . . .pssm file completed" % datetime.now())
    else:
        res = input("Would you like to download the PSIBLAST database? (~250GB, this is required to use your own sequence) [y/n]")
        while True:
            if res == "y":
                p.download_db()
                p.decompress_db()
                print("download complete, please re-run ProSPr")
                break
            if res == "n":
                break
            else:
                res = input("invalid, use 'y' or 'n':")

    print("[%s] building final pkl sequence." % datetime.now())
    b.join()
    s = Sequence(args.domain)
    s.build(args)
    print("[%s] pkl file ready, you can now use prospr run %s." % (datetime.now(), args.domain))


# detect database, if not present, start download
# psiblast -query $FASTAX -db $PSDB -out_ascii_pssm $BASE".pssm" -out $BASE".out" $PSOPTIONS
#FASTAX = args.domain +".x"
#BASE = savedir/domain

if __name__ == "__main__":
    args = parser.parse_args()
    if args.command == 'run':
        main(args)
    elif args.command == 'build':
        build(args)
