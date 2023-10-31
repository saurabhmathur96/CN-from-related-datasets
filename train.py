from os import path, makedirs
import argparse

from sklearn.naive_bayes import MultinomialNB
from tqdm import tqdm
import joblib

from structure_learning import learn_structure, learn_chow_liu_leaf
from structure_score import mi_score, bic_score2, group_bic_score2
from parameter_learning import estimate_parameters
from dataset import get_dataset





random_states = [2827, 5030,   16, 7891, 1762, 3043, 5060, 9312, 5468, 5321]


parser = argparse.ArgumentParser()
parser.add_argument("target")
parser.add_argument("n_samples", type=int)
args = parser.parse_args()

if args.target == "GDM":
    target = "GDM"
    dataset = "numom2b_gdm_race"
elif args.target == "PTB":
    target = "PTB"
    dataset = "numom2b_ptb_race"
else:
    raise ValueError("Invalid target")

train, test = get_dataset(dataset)
n_groups = train.r[train.group_col]

frac = train.as_df()[target].mean()
print(f"{frac:.4f}")
base = path.join("models", target)
n_samples = args.n_samples
assert n_samples <= 10

for i, train_ in tqdm(
        enumerate(train.bootstrap_samples(n_samples, random_states)),
        total=n_samples):
    sample_path = path.join(base, f"sample{i}")
    makedirs(sample_path, exist_ok=True)

    train_df = train_.as_df()
    clf = MultinomialNB().fit(
        train_df.drop(target, axis=1),
        train_.as_df()[target])
    joblib.dump(clf, path.join(sample_path, "model0.pkl"))

    cnet = learn_structure(train_, learn_chow_liu_leaf,
                           mi_score, min_score=0.01)
    parameters = [estimate_parameters(cnet, d) for d in train_.datasets]
    joblib.dump([cnet, parameters], path.join(sample_path, "model1.pkl"))

    cnet2 = learn_structure(train_, learn_chow_liu_leaf,
                            bic_score2, min_score=0.0)
    parameters2 = [estimate_parameters(cnet2, d) for d in train_.datasets]
    joblib.dump([cnet2, parameters2], path.join(sample_path, "model2.pkl"))

    cnet3 = learn_structure(train_, learn_chow_liu_leaf,
                            group_bic_score2, min_score=0.0)
    parameters3 = [estimate_parameters(
        cnet3, d, leaf_only=True) for d in train_.datasets]
    joblib.dump([cnet3, parameters3], path.join(sample_path, "model3.pkl"))
