import threading
import subprocess
import sys
import os
from prospr import pconf
import multiprocessing
from datetime import datetime
import requests

def hhdb_dl_present():
     return os.path.exists(pconf.basedir + \
     "hhblits/uniclust30_2018_08_hhsuite.tar.gz")

def hhdb_db_present():
    on_disk = os.listdir(pconf.basedir + "hhblits")
    file_count = len(on_disk)
    if file_count > 1:
        return True
    else:
        return False

def hhdb_unzip(filename):
    subprocess.run(['tar', '--skip-old-files', '-xzvf', filename, '-C', pconf.basedir+"hhblits"])
    print("unzip complete")


def hhdb_install():
    url = "http://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/uniclust30_2018_08_hhsuite.tar.gz"
    filename = pconf.basedir + "hhblits/uniclust30_2018_08_hhsuite.tar.gz"
    print("downloading uniclust30 for hhblits")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_length = int(r.headers.get('content-length'))
        dl_total = 0
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=4096):
                if chunk:
                    dl_total += len(chunk)
                    f.write(chunk)
                    done = int(50 * dl_total / total_length)
                    sys.stdout.write("\r[%s%s] %s / %s MB" % ('=' * done, ' ' * (50-done), round(dl_total/1024**2,1), round(total_length/1024**2,1)) )
                    sys.stdout.flush()
    print("Download complete, extracting. . .")
    hhdb_unzip(filename)

class BlitsAndPottsRunner(threading.Thread):
    def __init__(s, domain, **kwargs):
        threading.Thread.__init__(s)
        s.includePotts = True
        s.out_dir = pconf.basedir + domain + "/"
        s.fasta = s.out_dir + domain + ".fasta"
        s.domain = domain
        s.hhdb = pconf.basedir + "hhblits/uniclust30_2018_08/uniclust30_2018_08"
        n_threads = kwargs.pop("n_threads", 1)
        s.hhoptions = " -e 0.01 -n 3 -B 100000 -Z 100000 -maxmem 4.0 -v 0 -cpu %d"%n_threads
        # overwrites defaults if passed in
        s.__dict__.update(kwargs)

    def run(s):
        bin_dir = "/usr/local/bin/"
        hbin = bin_dir + "hhblits"
        hmake = bin_dir + "hhmake"
        hhmf = s.out_dir + s.domain + ".hhm"
        hhrf = s.out_dir + s.domain + ".hhr"
        a3mf = s.out_dir + s.domain + ".a3m"
        hhblitsCommand = [hbin, "-i", s.fasta, "-oa3m", a3mf, "-d",s.hhdb, "-o", hhrf]
        hhblitsCommand.extend(s.hhoptions.split(" "))
# naw, do subprocess and proper console outputs
        subprocess.run(hhblitsCommand)

        print("[%s] hhblits completed." % datetime.now())
        print("[%s] hhmake running." % datetime.now())
        hhmakeCommand = [hmake, '-i', a3mf, '-o', hhmf]
        subprocess.run(hhmakeCommand)
        print("[%s] hhmake completed." % datetime.now())

        if s.includePotts:
            print("[%s] potts running." % datetime.now())
            a2mf = s.out_dir + s.domain + ".a2m"
            matf = s.out_dir + s.domain + ".mat"
            reformatCmd = ["/hh-suite/scripts/reformat.pl", a3mf, a2mf]
            subprocess.run(reformatCmd)

            import plmDCA_asymmetric
            plmDCA_asymmetric.initialize_runtime(["-nodisplay"])
            p = plmDCA_asymmetric.initialize()
            p.plmDCA_asymmetric(a2mf, matf, multiprocessing.cpu_count(), 1, nargout=0)
            print("[%s] potts completed." % datetime.now())
