#FROM nvidia/opencl:devel-centos7
FROM nvidia/cuda:9.2-devel-centos7
MAINTAINER Todd Millecam <todd.millecam@gmail.com>

RUN yum makecache -y && \
     yum install -y epel-release wget cmake vim octave && \
     yum makecache -y && \
     yum install -y conda sudo python36-pip && \
     wget ftp://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/ncbi-blast-2.9.0+-1.x86_64.rpm && \
     yum localinstall -y ncbi-blast-2.9.0+-1.x86_64.rpm && \
     yum group install -y "Development Tools" && \
     git clone https://github.com/soedinglab/hh-suite.git && \
     mkdir -p hh-suite/build && \
     pip3 install torch ipython pyfiglet scipy requests==2.22.0
WORKDIR hh-suite/build
RUN cmake .. && \
    make && \
    make install && \
    mkdir -p /prosprdbs

COPY ./ /opt/
WORKDIR /opt/potts
RUN wget https://ssd.mathworks.com/supportfiles/downloads/R2018a/deployment_files/R2018a/installers/glnxa64/MCR_R2018a_glnxa64_installer.zip && \
    unzip MCR_R2018a_glnxa64_installer.zip && \
    ./install -mode silent -agreeToLicense yes && \
    python3 setup.py install && \
    rm /usr/local/MATLAB/MATLAB_Runtime/v94/bin/glnxa64/libexpat.so.1 && \
    cp /usr/lib64/libexpat.so.1 /usr/local/MATLAB/MATLAB_Runtime/v94/bin/glnxa64/libexpat.so.1 
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/MATLAB/MATLAB_Runtime/v94/runtime/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v94/bin/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v94/sys/os/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v94/extern/bin/glnxa64"

ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["-h","","","","",""]
#CMD ["/bin/bash"]
