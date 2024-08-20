from src import workdir, parse_model_parameter_file
from src.mcmc import Chain
import numpy as np

import os

exp_path = "./closure_test_point_JIMWLK.pkl"
model_par = "./IP_DIFF_JIMWLK_prior_range_delete_unused"
mymcmc_PCSK = Chain(expdata_path=exp_path, model_parafile=model_par, mcmc_path="./mcmc_PCSK_sep_emu_closure/chain.pkl")


emuPathList_PCSK = ["./emulator_set1_closure.pkl", "./emulator_set2_closure.pkl", "./emulator_set3_closure.pkl"]
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
