# ProSPr: Protein Structure Prediction
Wendy Billings, Bryce Hedelius, Todd Millecam, David Wingate, Dennis Della Corte   
Brigham Young University     

This repository contains an open-source protein distance prediction network, ProSPr, released under the MIT license.

The manuscript corresponding to this work is currently under preparation and will be released shortly.

The preprint associated with a PREVIOUS VERSION () is available here: https://www.biorxiv.org/content/10.1101/830273v2  

All data required to reproduce the work are publicly acessible at https://files.physics.byu.edu/data/prospr/

*************************************
WARNING: The 2TB of data associated with the PREVIOUS PROSPR VERSION currently hosted on the ftp server WILL BE REMOVED ON 1 JUNE 2021. If you would like future access to the files, please download a local copy.
*************************************

### Running ProSPr

After downloading the code, a conda environment with all required dependencies can be created by running    
`conda env create -f prospr-env.yml`   
Once activated, make a prediction using   
`python prospr.py path_to_a3m_input.a3m`   
For more information, run    
`python prospr.py -h`    
to print the help text.   

ProSPr is also available as a docker container...

Contact: dennis.dellacorte@byu.edu
