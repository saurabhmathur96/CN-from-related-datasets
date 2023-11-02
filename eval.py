from os import path
import argparse

import numpy as np
from tqdm import trange
import joblib
from sklearn.metrics import roc_auc_score
from dataset import get_dataset
from parameter_learning import set_parameters



def predict(cnet, target, data):
    p = np.zeros((len(data), 2))
    df = data.as_df()
    for i, (_, row) in enumerate(df.iterrows()):
        query1 = {c: row[c] for c in df.columns}
        query1[target] = 1
        numerator = cnet.marginal(query1)

        denominator = cnet.marginal(
            {c: row[c] for c in df.columns if c != target})
        
        p1 = numerator/denominator
        p[i, 1] = p1
        p[i, 0] = 1-p1
    return p


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
base = path.join("models", target)
n_samples = args.n_samples
assert n_samples <= 10
n_groups = train.r[train.group_col]

scores = np.zeros((5, n_groups, n_samples))
ascores = np.zeros((5, n_samples))
ll = np.zeros((5, n_groups, n_samples))
for i in trange(n_samples):
    clf = joblib.load(path.join(base, f"sample{i}", "model0.pkl"))
    
    # (0) Naive Bayes
    p1 = []
    y = []
    for g, test_ in enumerate(test.datasets):
        test_data = test_.as_df().drop(target, axis=1)
        test_labels = test_.as_df()[target]
        
        p1.append(clf.predict_proba(test_data)[:, 1])
        y.append(test_labels.to_numpy())

        scores[0, g, i] = roc_auc_score(test_labels, p1[-1])
        jlp = clf.predict_joint_log_proba(test_data)
        ll[0, g, i] = np.sum(jlp[np.arange(len(test_)), test_labels])

    ascores[0, i] = roc_auc_score(np.concatenate(y), np.concatenate(p1))
    
    # (1) Separate CN for each group
    cnets = joblib.load(
        path.join(base, f"sample{i}", "model1.pkl"))
    
    p1 = []
    y = []
    for g, test_ in enumerate(test.datasets):
        cnet = cnets[g]
        test_labels = test_.as_df()[target]

        p1.append(predict(cnet, target, test_)[:, 1])
        y.append(test_labels.to_numpy())

        scores[1, g, i] = roc_auc_score(test_labels, p1[-1])
        ll[1, g, i] = cnet(test_)

    ascores[1, i] = roc_auc_score(np.concatenate(y), np.concatenate(p1))
    
    # (2) Single CN
    [cnet2, default_parameters2, parameters2] = joblib.load(
        path.join(base, f"sample{i}", "model2.pkl"))

    p1 = []
    y = []
    cnet2 = set_parameters(cnet2, default_parameters2)
    for g, test_ in enumerate(test.datasets):
        test_labels = test_.as_df()[target]

        p1.append(predict(cnet2, target, test_)[:, 1])
        y.append(test_labels.to_numpy())

        scores[2, g, i] = roc_auc_score(test_labels, p1[-1])
        ll[2, g, i] = cnet2(test_)

    ascores[2, i] = roc_auc_score(np.concatenate(y), np.concatenate(p1))
    
    # (3) Single structure, different parameters
    p1 = []
    y = []
    for g, test_ in enumerate(test.datasets):
        cnet2 = set_parameters(cnet2, parameters2[g])
        test_labels = test_.as_df()[target]

        p1.append(predict(cnet2, target, test_)[:, 1])
        y.append(test_.as_df()[target].to_numpy())

        scores[3, g, i] = roc_auc_score(test_labels, p1[-1])
        ll[3, g, i] = cnet2(test_)
    ascores[3, i] = roc_auc_score(np.concatenate(y), np.concatenate(p1))

    # (4) Mixed-effects model
    [cnet3, parameters3] = joblib.load(
        path.join(base, f"sample{i}", "model3.pkl"))

    p1 = []
    y = []
    for g, test_ in enumerate(test.datasets):
        cnet3 = set_parameters(cnet3, parameters3[g], leaf_only=True)
        test_labels = test_.as_df()[target]

        p1.append(predict(cnet3, target, test_)[:, 1])
        y.append(test_.as_df()[target].to_numpy())

        scores[4, g, i] = roc_auc_score(test_labels, p1[-1])
        ll[4, g, i] = cnet3(test_)
    ascores[4, i] = roc_auc_score(np.concatenate(y), np.concatenate(p1))
    

for a in [ll, scores]:
    for g in range(n_groups):
        row = []
        for m in range(5):
            mean = np.mean(a[m, g, :])
            std = np.std(a[m, g, :])
            row.append(f"{mean:.4f} ± {std:.2f}")
        print (" | ".join(row))
    print ("\n")

print ("\n")

row = []
for m in range(5):
    mean = np.mean(ascores[m])
    std = np.std(ascores[m])
    row.append(f"{mean:.4f} ± {std:.2f}")
print (" | ".join(row))

row = []
for m in range(5):
    mean = np.mean(np.sum(ll[m], axis=0))
    std = np.std(np.sum(ll[m], axis=0))
    row.append(f"{mean:.4f} ± {std:.2f}")
print (" | ".join(row))
