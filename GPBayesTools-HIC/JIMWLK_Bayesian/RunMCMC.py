from src import workdir, parse_model_parameter_file
from src.emulator_BAND import EmulatorBAND
from src.mcmc import Chain

import dill
import os

exp_path = "./exp_data_JIMWLK.pkl"
model_par = "./IP_DIFF_JIMWLK_prior_range_delete_unused"
mymcmc = Chain(expdata_path=exp_path, model_parafile=model_par)

emuPathList = ["./emulator.pkl"]
mymcmc.loadEmulator(emuPathList)

os.environ["OMP_NUM_THREADS"] = "1"
# may have to: export RDMAV_FORK_SAFE=1 before running the code
n_effective=8000
n_active=4000
n_prior=16000
sample="tpcn"
n_max_steps=100
random_state=42

n_total = 100000
n_evidence = 100000

pool = 12

sampler = mymcmc.run_pocoMC(n_effective=n_effective, n_active=n_active,
                            n_prior=n_prior, sample=sample,
                            n_max_steps=n_max_steps, random_state=random_state,
                            n_total=n_total, n_evidence=n_evidence, pool=pool)
