# DeepCause

The rep contains code for our paper accepted in ICML Workshop 2022 on Spurious correlations, Invariance and Stability (SCIS): Causal Discovery using Model Invariance 
via Knockoffs by Wasim Ahmad, Maha Shadaydeh and Joachim Denzler.

- The work can be cited once published.

```
```


## Overview

We discover full causal graph in a multivariate nonlinear system by testing model invariance against Knockoffs-based interventional environments:
1. First we train deep network <img src="https://render.githubusercontent.com/render/math?math=f_i"> using data from observational environment <img src="https://render.githubusercontent.com/render/math?math=E_i">.
2. Then we expose the model to Knockoffs-based interventional environments <img src="https://render.githubusercontent.com/render/math?math=E_k">. 
3. For each pair variables {<img src="https://render.githubusercontent.com/render/math?math=z_i">, <img src="https://render.githubusercontent.com/render/math?math=z_j">} in nonlinear system, we test model invariance across environments. <img src="https://render.githubusercontent.com/render/math?math=E_i"> and <img src="https://render.githubusercontent.com/render/math?math=E_j">, use the classifier <img src="https://render.githubusercontent.com/render/math?math=f_i"> to partition <img src="https://render.githubusercontent.com/render/math?math=E_j">: <img src="https://render.githubusercontent.com/render/math?math=E_j = E_j^{i\checkmark} \cup E_j^{i\times}">,
where <img src="https://render.githubusercontent.com/render/math?math=D_i">: <img src="https://render.githubusercontent.com/render/math?math=D_k^{i\checkmark}"> contains examples that are predicted correctly and <img src="https://render.githubusercontent.com/render/math?math=E_j">: <img src="https://render.githubusercontent.com/render/math?math=E_j^{i\times}"> contains examples that are misclassified by <img src="https://render.githubusercontent.com/render/math?math=E_j">: <img src="https://render.githubusercontent.com/render/math?math=f_i">.
4. We perform KS test over distribution <img src="https://render.githubusercontent.com/render/math?math=D_i">, <img src="https://render.githubusercontent.com/render/math?math=D_j"> of model residuals in various environments. 
Our NULL hypothesis <img src="https://render.githubusercontent.com/render/math?math=H_o"> is that variable <img src="https://render.githubusercontent.com/render/math?math=z_i"> does not cause <img src="https://render.githubusercontent.com/render/math?math=z_j">, 
<img src="https://render.githubusercontent.com/render/math?math=D_i"> `?` <img src="https://render.githubusercontent.com/render/math?math=D_i">, else the alternate hypothesis <img src="https://render.githubusercontent.com/render/math?math=H_1">: <img src="https://render.githubusercontent.com/render/math?math=D_i"> = <img src="https://render.githubusercontent.com/render/math?math=D_i">  is accepted.


## Data
Datasets should be put under the directory `datasets/`.
We test our method on synethic as well as real data which can be found under `datasets/` directory. The real data we used is average daily river discharges that can be downloaded online.

### Quickstart
`.bin/` contains all the scripts for running the baselines and our algorithm.

## Code
`src/main.py` is our main file, where non-linear is data is modelled with deep networks.
- `src/deepcause.py` for actual and counterfactual outcome generation using interventions.
- `src/preprocessing.py` For data loading and preprocessing.
- `src/knockoffs.py` generate knockoffs of the original variables.
- `src/daignostics.py` to determine to goodness of the generated knockoff copies.
- `src/data/` contains the data pre-processing and loading pipeline for different datasets.
- `src/model/` contains trained models that we used for different datasets.

## Dependencies
`requirements.txt` contains all the packages that are related to the project.
To install them, simply create a new [conda](https://docs.conda.io/en/latest/) environment and type
```
pip install requirements.txt
```

## Acknowledgement

This work is funded by the Carl Zeiss Foundation within the scope of the program line "Breakthroughs: Exploring Intelligent Systems" for "Digitization — explore the basics, use applications" and the DFG grant SH 1682/1-1.