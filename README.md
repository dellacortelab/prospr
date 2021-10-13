# ProSPr: Protein Structure Prediction
Wendy Billings, Jacob Stern, Bryce Hedelius, Todd Millecam, David Wingate, Dennis Della Corte   
Brigham Young University     

This repository contains an open-source protein distance prediction network, ProSPr, released under the MIT license.

The manuscript corresponding to this work is currently under preparation and will be released shortly.

The preprint associated with a PREVIOUS VERSION (https://github.com/dellacortelab/prospr/tree/prospr1) is available here: https://www.biorxiv.org/content/10.1101/830273v2  

All data required to reproduce the work are publicly acessible at https://files.physics.byu.edu/data/prospr/

*************************************
WARNING: The 2TB of data associated with the PREVIOUS PROSPR VERSION currently hosted on the ftp server WILL BE REMOVED ON 1 JUNE 2021. If you would like future access to the files, please download a local copy.
*************************************

### Running ProSPr

After downloading the code, a conda environment with all required dependencies can be created by running    
```
conda env create -f dependencies/prospr-env.yml
```   
Once activated
```
# Make a prediction:
python3 prospr.py predict --a3m ./data/inputs/T1034.a3m
# Or train a new network
python3 prospr.py train
# Or evaluate an existing network
python3 prospr.py evaluate --a3m ./data/inputs/T1034.a3m --pdb ./data/inputs/T1034-D1.pdb
```
For more information, run    
```
python3 prospr.py -h
```
to print the help text.   


### Docker
Alternatively to conda, you can use Docker. To run the code in a Docker container, run the following after cloning this repository:
```
cd prospr
# Build the docker image
docker build -t prospr-dev dependencies
# Run a docker container interactively
docker run -it --name myname-prospr-dev --rm -v $(pwd):/code prospr-dev
# Then, inside the docker container, make a prediction:
cd code
python3 prospr.py predict --a3m ./data/inputs/T1034.a3m
# Or train a new network
python3 prospr.py train
# Or evaluate an existing network
python3 prospr.py evaluate --a3m ./data/inputs/T1034.a3m --pdb ./data/inputs/T1034-D1.pdb
```

Contact: dennis.dellacorte@byu.edu
