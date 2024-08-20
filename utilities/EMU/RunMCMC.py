from src import workdir, parse_model_parameter_file
from src.mcmc import Chain
import numpy as np

import os

exp_path = "./exp_data_JIMWLK_LOG.pkl"
model_par = "./IP_DIFF_JIMWLK_prior_range_delete_unused"
mymcmc_PCSK = Chain(expdata_path=exp_path, model_parafile=model_par, mcmc_path="./mcmc_PCSK_cov_diag/chain.pkl")

#mymcmc_Scikit = Chain(expdata_path=exp_path, model_parafile=model_par, mcmc_path="./mcmc_Scikit/chain.pkl")

#######
# add covariance matrix here:
# load the covariance matrix from Cov_matrix.txt
cov_matrix = np.loadtxt("Cov_matrix.txt")

# invert the covariance matrix
cov_matrix = np.linalg.inv(cov_matrix)

# for the first 7x7 indices in the covariance matrix, set the covariance to 0, 
# only take the diagonal part
# Remove the ALICE cov matrix
#for i in range(7):
#    for j in range(7):
#        if i != j:
#            cov_matrix[i][j] = 0.

# Remove the CMS cov matrix
#for i in range(7,13):
#    for j in range(7,13):
#        if i != j:
#            cov_matrix[j][i] = 0.

for i in range(0,13):
    for j in range(0,13):
        if i != j:
            cov_matrix[j][i] = 0.

mymcmc_PCSK.expdata_cov = cov_matrix

emuPathList_PCSK = ["./emulator.pkl"]
mymcmc_PCSK.loadEmulator(emuPathList_PCSK)

#emuPathList_Scikit = ["./emulator_scikit.pkl"]
#mymcmc_Scikit.loadEmulator(emuPathList_Scikit)

os.environ["OMP_NUM_THREADS"] = "1"
# may have to: export RDMAV_FORK_SAFE=1 before running the code
n_effective=8000
n_active=3600
n_prior=8000
sample="tpcn"
n_max_steps=100
random_state=42

n_total = 15000
n_evidence = 15000

pool = 12

sampler = mymcmc_PCSK.run_pocoMC(n_effective=n_effective, n_active=n_active,
                            n_prior=n_prior, sample=sample,
                            n_max_steps=n_max_steps, random_state=random_state,
                            n_total=n_total, n_evidence=n_evidence, pool=pool)

#sampler1 = mymcmc_Scikit.run_pocoMC(n_effective=n_effective, n_active=n_active,
#                            n_prior=n_prior, sample=sample,
#                            n_max_steps=n_max_steps, random_state=random_state,
#                            n_total=n_total, n_evidence=n_evidence, pool=pool)
