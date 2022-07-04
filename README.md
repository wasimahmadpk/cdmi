# DeepCause

The rep contains code for our paper accepted in ICML Workshop 2022 on Spurious correlations, Invariance and Stability (SCIS): Causal Discovery using Model Invariance 
via Knockoffs by Wasim Ahmad, Maha Shadaydeh and Joachim Denzler.

- If you find this work useful and use it on your own research, please cite our paper.

```
@inproceedings{bao2021predict,
  title={Predict then Interpolate: A Simple Algorithm to Learn Stable Classifiers},
  author={Bao, Yujia and Chang, Shiyu and Barzilay, Regina},
  booktitle={International Conference on Machine Learning},
  year={2021},
  organization={PMLR}
}
```


## Overview

We discover full causal graph in a nonlinear system by testing model invariance against Knockoffs-based interventional environments:
1. First we train deep network <img src="https://render.githubusercontent.com/render/math?math=f_i"> using data from observational environment <img src="https://render.githubusercontent.com/render/math?math=E_i">.
2. Then we expose the model to Knockoffs-based interventional environments <img src="https://render.githubusercontent.com/render/math?math=E_int">. For each pair of training environments <img src="https://render.githubusercontent.com/render/math?math=E_i"> and <img src="https://render.githubusercontent.com/render/math?math=E_j">, use the classifier <img src="https://render.githubusercontent.com/render/math?math=f_i"> to partition <img src="https://render.githubusercontent.com/render/math?math=E_j">: <img src="https://render.githubusercontent.com/render/math?math=E_j = E_j^{i\checkmark} \cup E_j^{i\times}">,
where <img src="https://render.githubusercontent.com/render/math?math=E_j">: <img src="https://render.githubusercontent.com/render/math?math=E_j^{i\checkmark}"> contains examples that are predicted correctly and <img src="https://render.githubusercontent.com/render/math?math=E_j">: <img src="https://render.githubusercontent.com/render/math?math=E_j^{i\times}"> contains examples that are misclassified by <img src="https://render.githubusercontent.com/render/math?math=E_j">: <img src="https://render.githubusercontent.com/render/math?math=f_i">.
3. Train the final model by minimizing the worst-case risk over all interpolations of the partitions.


## Data
Datasets should be put under the directory `datasets/`.
We ran experiments on a total of 4 datasets. MNIST and CelebA can be directly downloaded from the PyTorch API. For beer review and ASK2ME, due to liscense issues, you may contact me ([yujia@csail.mit.edu](yujia@csail.mit.edu)) for the processed data.

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