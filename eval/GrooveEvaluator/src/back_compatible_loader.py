# ------------------------------------------------------------------------------------------------
# ESSENTIAL FOR FINDING THE CORRECT PATH TO UNPICKLE OLDER PICKLED Evaluator Instances
# ------------------------------------------------------------------------------------------------
from eval.GrooveEvaluator import src
import wandb
import os, sys
import bz2
sys.modules['GrooveEvaluator'] = src
sys.modules['wandb'] = wandb
sys.modules['Metadata'] = dict

# ------------------------------------------------------------------------------------------------

import pickle

def load_evaluator(eval_path):
    if "bz2" in eval_path:
        ifile = bz2.BZ2File(os.path.join(eval_path), 'rb')
        data = pickle.load(ifile)
        ifile.close()
    else:
        data = pickle.load(open(eval_path, "rb"))
    return data


