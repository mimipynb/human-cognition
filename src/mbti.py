"""

    src/mbti.py

    MBTI chain with SVC and mean pooled vector embeddings as input features.
    Dataset used:
        - https://huggingface.co/datasets/kl08/myers-briggs-type-indicator
        ... Also only has `train` split.

        - pandalla/Machine_Mindset_MBTI_dataset
        ... Better for fine-tuning instruct llms

    Papers
    https://arxiv.org/pdf/2312.12999

"""

import joblib
import pandas as pd
from dataclasses import dataclass, field

from sklearn.svm import SVC
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, GridSearchCV

SEED = 0
data_card = "kl08/myers-briggs-type-indicator"
model_card = "sentence-transformers/all-MiniLM-L6-v2"
# data_card = "pandalla/Machine_Mindset_MBTI_dataset" jfc no labels smh

# SVC hyperparameters
param_grid = {
    'C': [1, 10, 50, 100],
    'kernel': ['linear', 'rbf', 'sigmoid'],
    'degree': [2, 3, 4, 5]
}

mbti_col = ['m1', 'm2', 'm3', 'm4']

@dataclass(slots=True)
class MBTIChain:
    models: list = field(default_factory=list)
    encoders: list = field(default_factory=list)

    def add_state(self, encoder, model):
        self.encoders += [encoder]
        self.models += [model]

    def fetch(self, inputs):
        """ Invoked during streaming. Input type must be already encoded. """

        for idx, model in enumerate(self.models):
            pred = model.predict(inputs)
            yield self.encoders[idx].inverse_transform(pred)

def fit_svc(X_train, Y_train, X_test, Y_test):
    """ Fits using SVC and returns the best model. """

    print('Training model now ....')
    base_lr = SVC(random_state=SEED)

    for idx, model_tag in enumerate(mbti_col):
        # Run SVC params
        clf = GridSearchCV(base_lr, param_grid, cv=5, return_train_score=False, verbose=True, scoring='accuracy')
        clf.fit(X_train, Y_train[:, idx])
        # Output Results
        print(f'MBTI ({idx}) - {model_tag} - train score: {clf.score(X_train, Y_train[:, idx])} \t test score: {clf.score(X_test, Y_test[:, idx])}')
        # Add to Dataclass Chain
        yield clf.best_estimator_

def load_mbti_dataset():
    """ Loads the models and fits the MBTI chain model. """

    # Loads the datset
    ds = load_dataset(data_card)

    # Preprocesses the dataset
    df = ds['train'].to_pandas()
    df[mbti_col] = df['type'].str.split('', expand=True).iloc[:, 1:5]

    if set(df.columns) != set(('posts', 'type', 'm1', 'm2', 'm3', 'm4')):
        raise ValueError(f"Columns mismatch. Expecting ('posts', 'type', 'm1', 'm2', 'm3', 'm4')")

    texts, features = df['posts'].values, df[mbti_col].values

    # Loading SentenceTransformers to encode the texts
    print(f"Loading sentence model now . . . .")
    model = SentenceTransformer(model_card)
    inputs = model.encode(texts)
    mbti_chain = MBTIChain()

    # Creating Label Encoders
    feats = {}
    labellers = []
    for item, label in enumerate(mbti_col):
        labeller = LabelEncoder().fit(features[:, item])
        output = labeller.transform(features[:, item])
        print(f"Iter ({item}) \t {output.shape}")
        feats[label] = output
        labellers += [labeller]

    y = pd.DataFrame(feats).values # this way ensures the cols and rows of the matrix are returned as expected
    X_train, X_test, Y_train, Y_test = train_test_split(
        inputs, y, train_size=0.70, random_state=SEED
    )

    for idx, model in enumerate(fit_svc(X_train, Y_train, X_test, Y_test)):
        mbti_chain.add_state(labellers[idx], model)

    return mbti_chain

def setup_mbti_chain(output_path: str):
    """ To setup smol MBTI chain model as prior states. """

    mbti_chain = load_mbti_dataset()
    joblib.dump(mbti_chain, output_path)
    print(f"MBTI chain model saved to: {output_path}")

if __name__ == "__main__":
    import os
    import sys
    import argparse
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from main import output_path  # Now this works

    print(f"Output folder path: {output_path}")
    parser = argparse.ArgumentParser(description="Setting up MBTI chain model")
    parser.add_argument("--output-path", action="store", type=str, help="Path to save the model", default="mbti_chain.joblib")

    args = parser.parse_args()

    if not args.output_path.endswith(".joblib"):
        args.output_path += ".joblib"
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    full_path = output_path / args.output_path

    try:
        setup_mbti_chain(full_path)
    except Exception as e:
        print(f"Error setting up model: {e}")
        raise
    else:
        print(f"MBTI Model ready to be used.")

