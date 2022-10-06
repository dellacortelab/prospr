# ProSPr: Protein Structure Prediction

![alt text](https://github.com/dellacortelab/prospr/blob/master/data/results/T1034/dist_pred_label.png?raw=true)

This repository contains an open-source protein distance prediction network, ProSPr, released under the MIT license.

### Running ProSPr

After downloading the code, a conda environment with all required dependencies can be created by running    
```
conda env create -f dependencies/prospr-env.yml
```   
Once activated
```
# Make a prediction:
python3 prospr.py predict --a3m ./data/evaluate/T1034.a3m

# Or train a new network
python3 prospr.py train

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
docker run -it --gpus all --name myname-prospr-dev --rm -v $(pwd):/code prospr-dev
# Then, inside the docker container, make a prediction:
cd code
python3 prospr.py predict --a3m ./data/evaluate/T1034.a3m
# Or train a new network
python3 prospr.py train
```

Contact: dennis.dellacorte@byu.edu

Wendy Billings, Jacob Stern, Bryce Hedelius, Todd Millecam, David Wingate, Dennis Della Corte   
Brigham Young University

The manuscript corresponding to this work is available here:
https://www.biorxiv.org/content/10.1101/2021.10.14.464472v1

The preprint associated with a PREVIOUS VERSION (https://github.com/dellacortelab/prospr/tree/prospr1) is available here: https://www.biorxiv.org/content/10.1101/830273v2  

A comparative study of ProSPr, trRosetta, and AlphaFold is available here:
Billings, W.M., Morris, C.J. & Della Corte, D. The whole is greater than its parts: ensembling improves protein contact prediction. Sci Rep 11, 8039 (2021). https://doi.org/10.1038/s41598-021-87524-0

All data required to reproduce the work are publicly acessible at https://files.physics.byu.edu/data/prospr/
