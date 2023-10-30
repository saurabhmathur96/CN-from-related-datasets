from dataset import get_dataset
from sklearn.metrics import mean_squared_error, roc_auc_score
from parameter_learning import set_parameters
import numpy as np
from tqdm import trange
import joblib
from os import path

def predict(cnet, target, data):
    p = np.zeros((len(data), 2))
    df = data.as_df()
    for i, (_, row) in enumerate(df.iterrows()):
        query1 = { c:row[c] for c in df.columns }
        assert "GDM" in query1
        query1[target] = 1
        numerator = cnet.marginal(query1)

        denominator = cnet.marginal({ c:row[c] for c in df.columns if c!=target })

        p1 = numerator/denominator
        p[i, 1] = p1
        p[i, 0] = 1-p1
    return p

train, test = get_dataset("numom2b_ptb_race")
target = "PTB"
base = path.join("models", target)
n_samples = 1
n_groups = train.r[train.group_col]

scores = np.zeros((4, n_groups, n_samples))
ll = np.zeros((4, n_groups, n_samples))
for i in trange(n_samples):
    clf = joblib.load(path.join(base, f"sample{i}", "model0.pkl"))
    [cnet, parameters] = joblib.load(path.join(base, f"sample{i}", "model1.pkl"))
    [cnet2, parameters2] = joblib.load(path.join(base, f"sample{i}", "model2.pkl"))
    [cnet3, parameters3] = joblib.load(path.join(base, f"sample{i}", "model3.pkl"))
    for g, test_ in enumerate(test.datasets):
        test_df = test_.as_df().drop(target,axis=1)
        scores[0, g, i] = roc_auc_score(test_.as_df()[target], clf.predict_proba(test_df)[:, 1] )

        cnet = set_parameters(cnet, parameters[g])
        scores[1, g, i] = roc_auc_score(test_.as_df()[target], predict(cnet, target, test_)[:, 1]) 
        ll[1, g, i] = cnet(test_)

        cnet2 = set_parameters(cnet2, parameters2[g])
        scores[2, g, i] = roc_auc_score(test_.as_df()[target], predict(cnet2, target, test_)[:, 1])
        ll[2, g, i] = cnet2(test_)

        cnet3 = set_parameters(cnet3, parameters3[g])
        scores[3, g, i] = roc_auc_score(test_.as_df()[target], predict(cnet3, target, test_)[:, 1]) 
        ll[3, g, i] = cnet3(test_)

for g in range(n_groups):
    m0, m1, m2, m3 = np.mean(scores[0, g, :]), np.mean(scores[1, g, :]),  np.mean(scores[2, g, :]),  np.mean(scores[3, g, :])
    s0, s1, s2, s3 = np.std(scores[0, g, :]), np.std(scores[1, g, :]), np.std(scores[2, g, :]), np.std(scores[3, g, :])
    print (f"{m0:.4f} ± {s0:.2f} | {m1:.4f} ± {s1:.2f} | {m2:.4f} ± {s2:.2f} | {m3:.4f} ± {s3:.2f}")

for g in range(n_groups):
    m0, m1, m2, m3 = np.mean(ll[0, g, :]), np.mean(ll[1, g, :]),  np.mean(ll[2, g, :]),  np.mean(ll[3, g, :])
    s0, s1, s2, s3 = np.std(ll[0, g, :]), np.std(ll[1, g, :]), np.std(ll[2, g, :]), np.std(ll[3, g, :])
    print (f"{m0:.4f} ± {s0:.2f} | {m1:.4f} ± {s1:.2f} | {m2:.4f} ± {s2:.2f} | {m3:.4f} ± {s3:.2f}")

