"""
    dialogue.py

    This file contains the functions to process the daily dialogue dataset:
    https://huggingface.co/datasets/li2017dailydialog/daily_dialog

    Setup file for the dialogue intensity model. The model predicts the intensity of the dialogue based on the act and emotion labels in the dataset.

    \text{Emote} =
    \begin{cases}
        0, & \text{no emote} \\
        1, & \text{expressive}
    \end{cases}

    \\ P(\text{Emote} \mid \text{Act}, \text{Dialog\_Size})

    NOTE: This doesn't consider the chat's topic or contextual information. This is really just lazy way (over kill) method of learning from the frequencies of user / agent's engagement + intentions. 

"""

import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from datasets import load_dataset, DatasetDict, concatenate_datasets

SEED = 42
data_card = "li2017dailydialog/daily_dialog"

# act and emote labels in the dataset
intent_map = ['no_intent', 'inform', 'question', 'directive', 'commisive']
emote_map = ['noemote', 'anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']

# newly added columns for the dataset. Contains the counts.
act_col = [f"act_{i}" for i in range(1, len(intent_map))]
emote_col = [f"emote_{i}" for i in range(len(emote_map))]
# features labels used for prediction
pred_cols = act_col + emote_col + ['dialog_size']

@dataclass(slots=True)
class DialogueEmotion:
    scaler: StandardScaler
    model: LogisticRegression
    label_map: dict

    def fetch(self, inputs):
        """ Invoked during streaming. """

        if inputs.shape != (1, self.model.n_features_in_):
            raise ValueError(f"Input features do not match the model. Insert {self.model.feature_names_in_ }. Only accept shape: (1, {self.model.n_features_in_})")

        x = self.scaler.transform(inputs)
        return dict(zip(self.label_map.values(), *self.model.predict_proba(x)))

def dialogue_intensity(ds):
    """ Preprocessing function for task: Dialogue intensity conditioned to dialogue.  """

    df = ds.to_pandas()
    dummies = pd.get_dummies(df.explode('act')['act'], prefix='act', dtype=int)
    e_dummies = pd.get_dummies(df.explode('emotion')['emotion'], prefix='emote', dtype=int)

    count_df = dummies.groupby(dummies.index).sum()
    e_count_df = e_dummies.groupby(e_dummies.index).sum()

    df = pd.concat([df, count_df], axis=1)
    df = pd.concat([df, e_count_df], axis=1)
    df['dialog_size'] = df['dialog'].str.len()

    if set(df.columns) != set(('dialog', 'act', 'emotion', 'act_1', 'act_2', 'act_3', 'act_4', 'emote_0', 'emote_1', 'emote_2', 'emote_3', 'emote_4', 'emote_5', 'emote_6', 'dialog_size')):
        raise ValueError("Columns not added correctly, received: ", df.columns)

    df[act_col] /= df['dialog_size'].values[:, None]
    df[emote_col] /= df['dialog_size'].values[:, None]
    # prepare for binary classification: emotionless vs emotional
    df = df.explode('emotion')
    df['emotion'] = np.where(df['emotion'] != 0, 1, 0)

    return df[pred_cols].values, df['emotion'].values

def fit_logistic(X_train, y_train, X_test, y_test):
    """ Fit a logistic regression model to predict dialogue intensity. """

    # 2. Apply SMOTE (Oversampling minority class in training set)
    smote = SMOTE(random_state=SEED)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # 3. Scale features AFTER splitting (to avoid data leakage)
    scaler = StandardScaler()
    X_train_resampled = scaler.fit_transform(X_train_resampled)
    X_test = scaler.transform(X_test)

    # 4. Train Model (Using class weights to handle imbalance)
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train_resampled), y=y_train_resampled)
    model = LogisticRegression(class_weight={0: class_weights[0], 1: class_weights[1]})
    model.feature_names_in_ = pred_cols
    model.fit(X_train_resampled, y_train_resampled)

    # 5. Evaluate Model
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return DialogueEmotion(scaler=scaler, model=model, label_map={0: "no_emotion", 1: "emotional"})

def setup_dialogue(output_path: str):
    """ Main loader for the dialogue intensity model. """

    ds = load_dataset(data_card)

    if not isinstance(ds, DatasetDict):
        raise ValueError("Dataset not loaded correctly")

    X_train, y_train = dialogue_intensity(concatenate_datasets([ds['train'], ds['validation']]))
    X_test, y_test = dialogue_intensity(ds['test'])
    model = fit_logistic(X_train, y_train, X_test, y_test)

    joblib.dump(model, output_path)
    print(f"Model saved to path: {output_path}!")

if __name__ == "__main__":

    import os
    import sys
    import argparse

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    from main import output_path  # Now this works

    print(f"Output folder path: {output_path}")
    parser = argparse.ArgumentParser(description="Process the daily dialogue dataset")
    parser.add_argument("--output-path", action="store", type=str, help="Path to save the model", default="dialogue_intensity.joblib")

    args = parser.parse_args()

    if not args.output_path.endswith(".joblib"):
        args.output_path += ".joblib"
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    full_path = output_path / args.output_path

    try:
        setup_dialogue(full_path)
    except Exception as e:
        print(f"Error setting up model: {e}")
        raise
    else:
        print(f"Model ready to be used.")

