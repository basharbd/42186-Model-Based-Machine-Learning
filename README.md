# Bayesian Movie Recommendation with Probabilistic Matrix Factorization

This repository contains the final project for **42186 Model-Based Machine Learning** at the Technical University of Denmark.

The project studies a Bayesian matrix factorization model for predicting MovieLens ratings. The main idea is to model each user with a latent preference vector and each movie with a latent attribute vector. The observed rating is generated from a global mean, user and movie bias terms, and the interaction between the corresponding latent vectors.

## Project goal

The project asks whether a Bayesian matrix factorization model can predict movie ratings better than simple baseline models while also providing uncertainty estimates.

The final model is implemented in **Pyro** and trained using **stochastic variational inference** with an `AutoNormal` guide.

## Repository structure

```text
.
├── data/              # MovieLens data files
├── figures/           # Figures generated from the notebook
├── results/           # Result tables generated from the notebook
├── notebook.ipynb     # Self-explanatory executed project notebook
├── report.pdf         # Final 6-page IEEE-style project report
└── README.md
```

## Contents

The notebook contains the full workflow:

- loading and preprocessing the MovieLens dataset
- selecting a manageable user-movie subset
- exploratory data analysis
- global mean baseline
- regularized user/movie bias baseline
- Bayesian matrix factorization model
- Pyro implementation and SVI training
- posterior predictive sampling
- RMSE and MAE evaluation
- uncertainty intervals
- posterior predictive checks
- discussion of limitations and possible extensions

## Model summary

For an observed rating row `n`, let `i_n` be the encoded user index and `j_n` be the encoded movie index.

The Bayesian matrix factorization model is:

```math
\mu \sim \mathcal{N}(3.5, 1^2)
```

```math
\sigma \sim \mathrm{HalfCauchy}(1)
```

```math
b_i^{(u)} \sim \mathcal{N}(0, 1),
\qquad
b_j^{(m)} \sim \mathcal{N}(0, 1)
```

```math
u_i \sim \mathcal{N}(0, I_K),
\qquad
v_j \sim \mathcal{N}(0, I_K)
```

```math
r_n \sim \mathcal{N}
\left(
\mu + b_{i_n}^{(u)} + b_{j_n}^{(m)} + u_{i_n}^{\top}v_{j_n},
\sigma^2
\right)
```

The latent dimension used in the final experiment is `K = 3`.

## Main results

The final test-set performance is:

| Model | RMSE | MAE |
|---|---:|---:|
| Bayesian matrix factorization | 0.817 | 0.619 |
| User/movie bias baseline | 0.831 | 0.632 |
| Global mean baseline | 0.972 | 0.768 |

The Bayesian model gives the best predictive performance. The improvement over the user/movie bias baseline is modest but consistent, which suggests that the latent interaction term captures additional user-movie structure beyond simple average effects.

## How to run

Create a Python environment and install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn torch pyro-ppl
```

Then open and run:

```text
notebook.ipynb
```

The notebook is designed to be run from top to bottom. It saves figures in the `figures/` folder and result tables in the `results/` folder.

## Final report

The final report is included as:

```text
report.pdf
```

It summarizes the project motivation, data preprocessing, model formulation, inference method, evaluation results, posterior predictive checks, limitations, future work, and declaration of generative AI use.

## Notes

The project uses a subset of MovieLens to keep the experiment reproducible and feasible to run on a standard laptop. The goal is not to build the strongest possible recommender system, but to demonstrate a complete model-based machine learning workflow: define a generative story, implement the probabilistic model, perform inference, evaluate predictions, and check posterior predictive behavior.
