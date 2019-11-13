from ftplib import FTP
from hashlib import md5
import os
from concurrent.futures import ThreadPoolExecutor
from prospr import pconf
import subprocess
#TODO make prospr call these functions and have it actually run the checks for psiblast
# add in the psiblast query parameters into the PsiblastManager, have it run as a non-blocking thread
def GZipCall(file_name):
    print("decompressing %s" % file_name)
    subprocess.run(['tar', '--skip-old-files', '-xzf', file_name])
 
class Reader:
    def __init__(self):
        self.data=""
    def __call__(self, s):
        try:
            self.data = s.decode('UTF-8').split()[0]
        except UnicodeDecodeError:
            self.data = ""
    
class PsiblastManager:
    """defaults:  local_dir="/data/psiblast", dl_server="ftp.ncbi.nlm.nih.gov", remote_dir="blast/db"
    can be changed, e.g. p = PsiblastManager(local_dir="/home/blast")"""
    def __init__(self, **kwargs):
        # defaults that can be overwritten
        self.local_dir= pconf.basedir + "psiblast/" 
        self.dl_server="ftp.ncbi.nlm.nih.gov"
        self.remote_dir="blast/db/"
        self.psbin = "/usr/bin/psiblast"
        # makes any named argument a variable on this object
        self.__dict__.update(kwargs) 

    def connect(self):
        self.ftp = FTP(self.dl_server)
        self.ftp.login("anonymous","")
        self.ftp.cwd(self.remote_dir)
        self.complete_file_list = self.ftp.nlst()
        self.nrfiles = [i for i in self.complete_file_list if i.startswith("nr")]

    def simple_check_db(self):
        """Checks to see if database is on the hard drive"""
        # first, is the volume mounted
        if len(os.popen("mount | grep \""+ pconf.basedir.split("/")[1] +"\"").readlines()) < 1:
            raise Exception("no data volume mounted") 
        # and if we pass that, see if the nr files exist
        on_disk = os.listdir(self.local_dir)
        if len([i for i in on_disk if i.startswith('nr')]) > 50:
            return True
        else:
            self.connect()
        db_files = [i for i in self.nrfiles if not i.endswith('md5')]
        diffs = list(set(db_files).difference(set(on_disk)))
        for fname in diffs:
            if fname.startswith("nr"):
                return False
        return True

    def full_check_db(self, count=1):
        """Compares data on the hard drive to checksums on the remote server, will check <count> number of files"""
        nr = self.nrfiles.copy()
        matches = {} 
        if count == "all":
            count = len([i for i in nr if not i.endswith('md5')])
        while count > 0:
            f = nr.pop()
            if not f.endswith(".md5"):
                with open(self.local_dir + f, 'rb') as mdf:
                    local_md5 = md5(mdf.read()).hexdigest()
                r = Reader()
                self.ftp.retrbinary('RETR %s' % f, r)
                if r.data == local_md5:
                    matches[f] = True
                else:
                    matches[f] = False
                count -= 1
        return matches

    def download_db(self):
        dlfiles = [i for i in self.nrfiles if not i.endswith("md5")]
        for f in dlfiles:
            print("downloading %s from %s" % (f, self.dl_server))
            self.ftp.retrbinary('RETR %s' % f, open(self.local_dir + f, 'wb').write)
    def decompressed(self):
        on_disk = os.listdir(self.local_dir)
        d_files = [i for i in on_disk if not i.endswith('.gz')]
        if len(d_files) > 15: #arbitrary really, not solid logic
            return True
        return False
 
    def decompress_db(self):
        on_disk = os.listdir(self.local_dir)
        zips = [i for i in on_disk if i.startswith('nr') and i.endswith('.gz')]
        os.chdir(self.local_dir)
        with ThreadPoolExecutor(max_workers=5) as ex:
            ex.map(GZipCall, zips)

    def build_query(self, domain, n_threads=1):
        dpath = pconf.basedir + domain + "/"
# check to see if pssm file is there and if they want to use it
        if os.path.exists(dpath+ domain +".pssm"):
            while True:
                resp = input("pssm file exists, would you like to use it? [y,n] ")
                if resp == "y":
                    return
                if resp == "n":
                    break 
        cmd = [self.psbin, 
         '-query', dpath + domain+".fasta",
         '-db', self.local_dir + "nr", 
         "-out_ascii_pssm", dpath + domain +".pssm",
         "-out", dpath + domain + ".out",
         "-evalue", "0.001",
         "-num_iterations", "3",
         "-save_pssm_after_last_round"]
        if n_threads > 1:
            cmd.extend(['-num_threads', '%d'%n_threads])
        subprocess.run(cmd)
