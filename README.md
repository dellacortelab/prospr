# ProSPr: Protein Structure Prediction

ProSPr distance map prediction (above diagonal) vs. ground truth (below diagonal) for CASP14 target T1034.

<img src="https://github.com/dellacortelab/prospr/blob/master/data/results/T1034/dist_pred_label.png?raw=true" alt="drawing" width="350"/>

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
[Evaluation of Deep Neural Network ProSPr for Accurate Protein Distance Predictions on CASP14 Targets](https://www.mdpi.com/1422-0067/22/23/12835)

To cite:
```
@Article{ijms222312835,
AUTHOR = {Stern, Jacob and Hedelius, Bryce and Fisher, Olivia and Billings, Wendy M. and Della Corte, Dennis},
TITLE = {Evaluation of Deep Neural Network ProSPr for Accurate Protein Distance Predictions on CASP14 Targets},
JOURNAL = {International Journal of Molecular Sciences},
VOLUME = {22},
YEAR = {2021},
NUMBER = {23},
ARTICLE-NUMBER = {12835},
URL = {https://www.mdpi.com/1422-0067/22/23/12835},
PubMedID = {34884640}
```
