"""
Returns the stochastic similarity for all fo the single 
agent policies defined (the ones trained in run_all.py)

"""

import os
import sys 
import tqdm

sys.path.append("..")
from os.path import join
from utils.sto_sim import Stochastic_Similarity
from itertools import product

# model names and reward types
model_names = list(product(
    ["full", "random", "frequent", "weighted", "cont"],
    ['0', '1', '2', '3']
))
models_path = "../models"

    
# write the results to a file
f = open("test.txt", 'w')

# initialize stochastic measure
sto = Stochastic_Similarity(None, None)

#iterate over all combinations
for i in tqdm.trange(len(model_names)):
    name = '_'.join(model_names[i])
    run_path = join(models_path, name)
    sto.configs = join(run_path, "configs_dict.pkl")

    # assumes best model only
    sto.model = join(run_path, "best_model.pkl")
    score = sto.similarity_score(50, 25)

    f.write(name + " ::: score ::: " + "{}".format(score) + '\n')        
f.close()

        
