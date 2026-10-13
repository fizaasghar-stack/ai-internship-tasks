"""
Task 2 (IMPROVED VERSION): Multi-Class Text Sentiment Classifier
Improvements over v1:
  1. Better preprocessing: contractions fixed (don't -> do not) + negation handling (not good -> not_good)
  2. Richer features: word TF-IDF (1-2 grams) + character TF-IDF (2-5 grams), sublinear TF
  3. Hyper-parameter tuning with GridSearchCV optimised directly for Macro F1
  4. Final model saved + normalised confusion matrix + error analysis

Run:  py sentiment_classifier_v2.py      (needs Tweets.csv in the same folder)
"""
import re
import joblib
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

BASELINE_MACRO_F1 = 0.7248   # v1 result (Logistic Regression)

# ------------------------------------------------------------- 1. LOAD DATA
df = pd.read_csv("Tweets.csv")[["text", "airline_sentiment"]]
df.columns = ["text", "label"]
df = df.dropna(subset=["text", "label"]).drop_duplicates(subset="text").reset_index(drop=True)
print("Class distribution:\n", df["label"].value_counts(), "\n")

# --------------------------------------------------------- 2. PREPROCESSING
NEGATIONS = {"not", "no", "never", "nor", "cannot", "neither"}
stop_words = set(stopwords.words("english")) - NEGATIONS
lemmatizer = WordNetLemmatizer()


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"@\w+", " ", text)                    # @mentions
    text = text.replace("#", " ")                        # hashtag sign
    text = text.replace("won't", "will not").replace("can't", "can not").replace("cant", "can not")
    text = re.sub(r"n't\b", " not", text)                # don't -> do not
    text = re.sub(r"[^a-z\s]", " ", text)                # punctuation, numbers, emojis

    out, negate = [], False
    for tok in text.split():
        if tok in NEGATIONS:                             # keep negation words
            negate = True
            out.append("not")
            continue
        if tok in stop_words or len(tok) < 2:            # stop-word removal
            continue
        tok = lemmatizer.lemmatize(lemmatizer.lemmatize(tok), pos="v")   # lemmatization
        out.append("not_" + tok if negate else tok)      # "not good" -> "not_good"
        negate = False
    return " ".join(out)


df["clean_text"] = df["text"].astype(str).apply(preprocess)
df = df[df["clean_text"].str.strip() != ""]

# ------------------------------------------------------- 3. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"],
)

# --------------------------------------------- 4. FEATURES + MODEL + TUNING
features = FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                             sublinear_tf=True, max_features=60000)),
])

pipe = Pipeline([
    ("features", features),
    ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
])

param_grid = [
    {"clf": [LogisticRegression(max_iter=2000, class_weight="balanced")],
     "clf__C": [1, 3, 10]},
    {"clf": [LinearSVC(class_weight="balanced")],
     "clf__C": [0.1, 0.3, 1]},
]

print("Tuning models with 3-fold cross validation (is me kuch minute lag sakte hain)...")
search = GridSearchCV(
    pipe, param_grid, scoring="f1_macro",
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    n_jobs=-1, verbose=1,
)
search.fit(X_train, y_train)

best_model = search.best_estimator_
print("\nBest settings:", search.best_params_)
print(f"Best cross-validation Macro F1: {search.best_score_:.4f}")

# ------------------------------------------------------------ 5. EVALUATION
preds = best_model.predict(X_test)
macro_f1 = f1_score(y_test, preds, average="macro")
print("\n=== FINAL TEST RESULTS ===")
print(classification_report(y_test, preds, digits=3))
print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
print(f"Macro F1 : {macro_f1:.4f}   (v1 baseline: {BASELINE_MACRO_F1:.4f}, "
      f"change: {macro_f1 - BASELINE_MACRO_F1:+.4f})")

# --------------------------------------------------- 6. CONFUSION MATRICES
labels = sorted(df["label"].unique())
cm = confusion_matrix(y_test, preds, labels=labels)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axes[0])
axes[0].set_title("Confusion Matrix (counts)")
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", xticklabels=labels, yticklabels=labels, ax=axes[1])
axes[1].set_title("Confusion Matrix (row-normalised)")
for ax in axes:
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_v2.png", dpi=150)

# ------------------------------------------------------ 7. ERROR ANALYSIS
errors = pd.DataFrame({
    "tweet": df.loc[X_test.index, "text"].values,
    "actual": y_test.values,
    "predicted": preds,
})
errors = errors[errors["actual"] != errors["predicted"]]
errors.head(50).to_csv("misclassified_examples.csv", index=False)
print(f"\nSaved {min(50, len(errors))} misclassified examples to misclassified_examples.csv")

# ----------------------------------------------------------- 8. SAVE MODEL
joblib.dump(best_model, "sentiment_model.joblib")


def predict_sentiment(sentence: str) -> str:
    return best_model.predict([preprocess(sentence)])[0]


print("\n=== Demo predictions ===")
for s in ["I love this product, it works great!",
          "Worst service ever, totally disappointed.",
          "The package arrived on Tuesday.",
          "The flight was not good at all.",
          "Thanks, the crew was not rude, really helpful!"]:
    print(f"{s}  ->  {predict_sentiment(s)}")
