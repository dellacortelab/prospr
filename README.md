# ProSPr: Protein Structure Prediction
Wendy M. Billings, Bryce Hedelius, Todd Millecam, David Wingate, Dennis Della Corte   
Brigham Young University     

This repository currently contains a democratized implementation of the AlphaFold distance prediction network.  
Please note that it is released under the LGPL-3.0 license.

The associated publication is currently under review, and can be found here on biorXiv: https://www.biorxiv.org/content/10.1101/830273v2   

ProSPr is available as a Docker container. After installing Docker, run:   
`docker run prospr/prospr`  
To build your own docker container after modifying this source code, run:   
`docker build .`   

All of the data used to train the ProSPr models (~2TB) is publically accessible at https://files.physics.byu.edu/data/prospr/   

All files are hosted on our ftp server including code dependencies: https://files.physics.byu.edu/data/prospr/

If you have difficulty using plmDCA, then you can download a pre-compiled python package for it.  It will require matlab 2018a or matlab redistributable 2018a to run (redistributable is free to download). The following command will download the installer and all data associated with plmDCA (linux and unix systems):

wget --recursive --reject "index.html" https://files.physics.byu.edu/data/prospr/potts-code/


Author contact: dennis.dellacorte@byu.edu


## CASP13 Comparison
Comparison of ProSPr and AlphaFold distance predictions on selected CASP13 target sequences, per label availability from published PDB structures:

![Alt text](images/T0954.jpeg?raw=true "T0954")
![Alt text](images/T0955.jpeg?raw=true "T0955")
![Alt text](images/T0957s1.jpeg?raw=true "T0957s1")
![Alt text](images/T0957s2.jpeg?raw=true "T0957s2")
![Alt text](images/T0958.jpeg?raw=true "T0958")
![Alt text](images/T0960.jpeg?raw=true "T0960")
![Alt text](images/T0963.jpeg?raw=true "T0963")
![Alt text](images/T0968s1.jpeg?raw=true "T0968s1")
![Alt text](images/T0968s2.jpeg?raw=true "T0968s2")
![Alt text](images/T0969.jpeg?raw=true "T0969")
![Alt text](images/T0980s1.jpeg?raw=true "T0980s1")
![Alt text](images/T0980s2.jpeg?raw=true "T0980s2")
![Alt text](images/T0986s1.jpeg?raw=true "T0986s1")
![Alt text](images/T1000.jpeg?raw=true "T1000")
![Alt text](images/T1003.jpeg?raw=true "T1003")
![Alt text](images/T1006.jpeg?raw=true "T1006")
![Alt text](images/T1009.jpeg?raw=true "T1009")
![Alt text](images/T1014.jpeg?raw=true "T1014")
![Alt text](images/T1016.jpeg?raw=true "T1016")
![Alt text](images/T1018.jpeg?raw=true "T1018")
![Alt text](images/T1021s1.jpeg?raw=true "T1021s1")
![Alt text](images/T1021s2.jpeg?raw=true "T1021s2")

AlphaFold predictions: https://github.com/deepmind/deepmind-research/tree/master/alphafold_casp13, download link under "Data"

PDB structure labels: https://predictioncenter.org/casp13/domains_summary.cgi

*(white indicates regions where calculating the label was not possible, eg. missing residues)*

For consistency in visualization across predictions made for different bin ranges, all distances (including labels) were rebinned into the 10 bin format defined for CASP14 distance predictions, found here https://predictioncenter.org/casp14/index.cgi?page=format#RR 

