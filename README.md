# ProSPr: Protein Structure Prediction
Wendy M. Billings, Bryce Hedelius, Todd Millecam, David Wingate, Dennis Della Corte   
Brigham Young University     

This repository currently contains a democratized implementation of the AlphaFold distance prediction network.  
Please note that the code is released under the LGPL-3.0 license.

The associated publication is currently under review, and can be found here on biorXiv: https://www.biorxiv.org/content/10.1101/830273v1   

ProSPr is available as a Docker container. After installing Docker, run:   
`docker run prospr/prospr`  
To build your own docker container after modifying this source code, run:   
`docker build .`   

All of the data used to train the ProSPr models (~2TB) is publically accessible at https://files.physics.byu.edu/data/prospr/   

All files are hosted on our ftp server including code dependencies: https://files.physics.byu.edu/data/prospr/

If you have difficulty using plmDCA, then you can download a pre-compiled python package for it.  It will require matlab 2018a or matlab redistributable 2018a to run (redistributable is free to download). The following command will download the installer and all data associated with plmDCA (linux and unix systems):

wget --recursive --reject "index.html" https://files.physics.byu.edu/data/prospr/potts-code/


Author contact: dennis.dellacorte@byu.edu
