from src import workdir, parse_model_parameter_file
from src.mcmc import Chain
import numpy as np

import os

#exp_path = "./exp_data_JIMWLK.pkl"
exp_path = "./exp_data_JIMWLK_no_tdiff.pkl"

model_par = "./IP_DIFF_JIMWLK_prior_range_delete_unused"
mymcmc_PCSK = Chain(expdata_path=exp_path, model_parafile=model_par, mcmc_path="./mcmc_PCSK_sep_emu_notdiff_expcov/chain.pkl")


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

#for i in range(0,13):
#    for j in range(0,13):
#        if i != j:
#            cov_matrix[j][i] = 0.

# remove the last 15 rows and columns in the covariance matrix
#cov_matrix = cov_matrix[:-15,:-15]

mymcmc_PCSK.expdata_cov = cov_matrix

emuPathList_PCSK = ["./emulator_set1.pkl", "./emulator_set2.pkl", "./emulator_set3.pkl"]
mymcmc_PCSK.loadEmulator(emuPathList_PCSK)


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
