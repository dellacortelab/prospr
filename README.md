# ProSPr: Protein Structure Prediction
Wendy M. Billings, Bryce Hedelius, Todd Millecam, David Wingate, Dennis Della Corte   
Brigham Young University     

This repository currently contains a democratized implementation of the AlphaFold distance prediction network.       

The associated publication is currently under review, and can be found here on biorXiv: https://www.biorxiv.org/content/10.1101/830273v1   

ProSPr is available as a Docker container. After installing Docker, run:   
`docker run prospr/prospr`  
To build your own docker container after modifying this source code, run:   
`docker build .`   

All of the data used to train the ProSPr models (~2TB) is publically accessible at https://files.physics.byu.edu/data/prospr/   

Author contact: dennis.dellacorte@byu.edu
