from eval.InterIntraEvaluator.src.utils import *
import pickle, bz2
import os

class InterIntraEvaluator:
    def __init__(self, groove_evaluator_sets, new_names=None, ignore_keys=None):
        """
        :param groove_evaluator_sets: dictionary of GrooveEvaluator objects to compare, example:
               sets_evals = {
                                "groovae":
                                    GrooveEvaluator(...),
                                "rosy":
                                    GrooveEvaluator(...),
                                "hopeful":
                                    GrooveEvaluator(...),
                            }

        :param new_names: dictionary of new names for the GrooveEvaluators
               new_names = {
                                "rosy": "Model 1",
                                "hopeful": "Model 2",
                                "groovae": "GrooVAE"
                            }
        :param ignore_keys: list of keys to ignore when comparing the GrooveEvaluators
                        (["Statistical::NoI", "Statistical::Total Step Density", "Statistical::Avg Voice Density",
                        "Statistical::Lowness", "Statistical::Midness", "Statistical::Hiness",
                        "Statistical::Vel Similarity Score", "Statistical::Weak to Strong Ratio",
                        "Syncopation::Lowsync", "Syncopation::Midsync", "Syncopation::Hisync",
                        "Syncopation::Lowsyness", "Syncopation::Midsyness", "Syncopation::Hisyness",
                        "Syncopation::Complexity"])
        """

        # ======================== INITIALIZATION ========================
        # =========     Static/Picklaable Attributes
        self.new_names = new_names
        self.ignore_keys = ignore_keys
        self.sets_evals = dict((new_names[key], value) for (key, value) in groove_evaluator_sets.items()) \
            if self.new_names is not None else groove_evaluator_sets

        # =========     Dynamically Constructed Attributes
        self.feature_sets = dict()
        self.create_allowed_features()

    # ================================================================
    # =========     Pickling Methods
    # ================================================================
    def __getstate__(self):
        state = {
            'new_names': self.new_names,
            'ignore_keys': self.ignore_keys,
            'sets_evals': self.sets_evals,
        }
        return state

    def __setstate__(self, state):
        self.new_names = state['new_names']
        self.ignore_keys = state['ignore_keys']
        self.sets_evals = state['sets_evals']
        self.create_allowed_features()

    def create_allowed_features(self):
        # attributes to dynamically construct using the static data that are picklable
        # (to use in constructor and __setstate__)
        self.feature_sets = {
            "GT": flatten_subset_genres(get_gt_feats_from_evaluator(list(self.sets_evals.values())[0]))}
        self.feature_sets.update({
            set_name: flatten_subset_genres(get_pd_feats_from_evaluator(eval)) for (set_name, eval) in
            self.sets_evals.items()
        })

        allowed_analysis = list(self.feature_sets['GT'].keys())
        # remove ignored keys from allowed analysis
        if self.ignore_keys is not None:
            for key in self.ignore_keys:
                allowed_analysis.remove(key) if key in allowed_analysis else print(
                    f"Can't ignore requested Key [`{key}`] as is not found in the feature set")

        for set_name in self.feature_sets.keys():
            for key in list(self.feature_sets[set_name].keys()):
                if key not in allowed_analysis:
                    self.feature_sets[set_name].pop(key)

    def dump(self, path):
        # check path ends with .bz2IIEval
        if not path.endswith(".bz2IIEval"):
            path += ".bz2IIEval"

        # make sure path is a valid path
        os.makedirs(os.path.dirname(path), exist_ok=True)

        with bz2.BZ2File(path, 'w') as f:
            pickle.dump(self, f)


def load_inter_intra_evaluator(path):
    with bz2.BZ2File(path, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
    from eval.GrooveEvaluator.src.evaluator import load_evaluator

    # prepare input data
    eval_1 = load_evaluator("path/test_set_full_fname.Eval.bz2")
    eval_2 = load_evaluator("path/test_set_full_fname.Eval.bz2")
    groove_evaluator_sets = { "groovae": eval_1, "rosy": eval_2}
    new_names = {"groovae": "GrooVAE",  "rosy": "Model 1"}
    ignore_keys = ["Statistical::NoI", "Statistical::Total Step Density", "Statistical::NEWWWWW"]

    # construct InterIntraEvaluator
    iiEvaluator = InterIntraEvaluator(
        groove_evaluator_sets= groove_evaluator_sets, new_names=new_names, ignore_keys=ignore_keys)

    # dump InterIntraEvaluator
    iiEvaluator.dump("testers/eval/misc/inter_intra_evaluator.bz2IIEval")

    # load InterIntraEvaluator
    iiEvaluator = load_inter_intra_evaluator("testers/eval/misc/inter_intra_evaluator.bz2IIEval")