"""
Task 2: Intelligent Multi-Class Text Sentiment Classifier
Pipeline: load data -> clean (stop-words + lemmatization) -> TF-IDF -> model -> F1 evaluation

Install:  pip install pandas scikit-learn nltk matplotlib seaborn
Dataset:  CSV with two columns: "text" and "label"
          (e.g. Kaggle "Twitter US Airline Sentiment": columns airline_sentiment, text)
"""
import re
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score, confusion_matrix

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ---------------------------------------------------------------- 1. LOAD DATA
# Apna dataset yahan load karo. Example (airline tweets):
# df = pd.read_csv("Tweets.csv")[["text", "airline_sentiment"]]
# df.columns = ["text", "label"]
df = pd.read_csv("Tweets.csv")[["text", "airline_sentiment"]]
df.columns = ["text", "label"]
df = df.dropna(subset=["text", "label"]).drop_duplicates(subset="text")
print("Class distribution:\n", df["label"].value_counts(), "\n")

# ------------------------------------------------------------- 2. PREPROCESSING
# "not", "no" jaisay negation words sentiment ke liye important hain, isliye keep kiye
stop_words = set(stopwords.words("english")) - {"not", "no", "nor", "never"}
lemmatizer = WordNetLemmatizer()


def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # URLs
    text = re.sub(r"@\w+|#", " ", text)           # mentions, hashtag sign
    text = re.sub(r"[^a-z\s]", " ", text)         # punctuation, numbers
    tokens = [t for t in text.split() if t not in stop_words and len(t) > 1]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]   # running -> running/run
    tokens = [lemmatizer.lemmatize(t, pos="v") for t in tokens]
    return " ".join(tokens)


df["clean_text"] = df["text"].astype(str).apply(preprocess)

# ----------------------------------------------------------- 3. TRAIN/TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"],
)

# ---------------------------------------------------- 4. TF-IDF + MODELS + EVAL
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Linear SVM": LinearSVC(class_weight="balanced"),
}

best_name, best_f1, best_pipe = None, 0, None
for name, clf in models.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=20000, min_df=2)),
        ("clf", clf),
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    macro_f1 = f1_score(y_test, preds, average="macro")
    print(f"=== {name} ===  Macro F1: {macro_f1:.4f}")
    print(classification_report(y_test, preds, digits=3))
    if macro_f1 > best_f1:
        best_name, best_f1, best_pipe = name, macro_f1, pipe

print(f"Best model: {best_name} (Macro F1 = {best_f1:.4f})")

# ------------------------------------------------------- 5. CONFUSION MATRIX
preds = best_pipe.predict(X_test)
labels = sorted(df["label"].unique())
cm = confusion_matrix(y_test, preds, labels=labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150)

# ------------------------------------------------------------ 6. TRY NEW TEXT
def predict_sentiment(sentence: str) -> str:
    return best_pipe.predict([preprocess(sentence)])[0]


for s in ["I love this product, it works great!",
          "Worst service ever, totally disappointed.",
          "The package arrived on Tuesday."]:
    print(f"{s}  ->  {predict_sentiment(s)}")
