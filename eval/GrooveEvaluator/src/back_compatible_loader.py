# ------------------------------------------------------------------------------------------------
# ESSENTIAL FOR FINDING THE CORRECT PATH TO UNPICKLE OLDER PICKLED Evaluator Instances
# ------------------------------------------------------------------------------------------------
from eval.GrooveEvaluator import src
import sys
sys.modules['GrooveEvaluator'] = src
# ------------------------------------------------------------------------------------------------

import pickle

def load_evaluator(eval_path):
    return pickle.load(open(eval_path, "rb"))


