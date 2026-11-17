import numpy as np, librosa, glob
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

SR, WIN = 9600, 256                          # Uno's analogRead runs ~9.6 kHz
CLASSES = ["traffic", "ambulance", "firetruck"]

def pitch(w):                                # zero-crossing pitch estimate
    s = np.sign(w - w.mean())
    return np.sum(s[1:] != s[:-1]) * SR / (2 * len(w))

def features(chunk):                         # 1 second of audio
    p = np.array([pitch(chunk[i:i+WIN]) for i in range(0, len(chunk)-WIN, WIN)])
    return [p.min(), p.max(), p.mean(), np.abs(np.diff(p)).mean()]

X, y = [], []
for label, name in enumerate(CLASSES):
    for f in glob.glob(f"data/{name}/*.wav"):
        sig, _ = librosa.load(f, sr=SR)
        for i in range(0, len(sig) - SR, SR):
            X.append(features(sig[i:i+SR])); y.append(label)
            
X, y = np.array(X), np.array(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=1)
tree = DecisionTreeClassifier(max_depth=4, random_state=1).fit(Xtr, ytr)
print(classification_report(yte, tree.predict(Xte), target_names=CLASSES))
print(confusion_matrix(yte, tree.predict(Xte)))
print(export_text(tree, feature_names=["pmin", "pmax", "pmean", "sweep"]))