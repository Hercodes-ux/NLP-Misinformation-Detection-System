import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

# --- 1. DATA LOADING & OPTIMIZED SAMPLING ---
# Load datasets
true_df = pd.read_csv('data/True.csv')
fake_df = pd.read_csv('data/Fake.csv')

# Assign labels
true_df['label'] = 1
fake_df['label'] = 0

# Merge
df = pd.concat([true_df, fake_df], ignore_index=True)

# OPTIMIZATION: Sampling 10,000 articles to prevent MemoryError
# This provides enough data for 99% accuracy while being RAM-friendly
df = df.sample(n=10000, random_state=42).reset_index(drop=True)

# Combine title and text
df['content'] = df['title'] + " " + df['text']
df['content'] = df['content'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

# --- 2. VISUALIZATION: WORD CLOUD ---
# We use a subset of the sampled data for the wordcloud to ensure speed
print("Generating WordCloud...")
wordcloud_text = ' '.join(df['content'].sample(n=2000, random_state=42))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("Key Indicators in News Articles")
plt.savefig("wordcloud.png") # For GitHub
plt.show()

# --- 3. PIPELINE SETUP ---
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['label'], test_size=0.2, random_state=42)

# Optimization: Added 'max_features' to prevent a giant, memory-heavy matrix
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
    ('lr', LogisticRegression())
])

# Optimization: n_jobs=1 to prevent 'TerminatedWorkerError'
lr_params = {
    'lr__C': [0.1, 1, 10],
    'lr__solver': ['liblinear']
}

print("Starting Logistic Regression GridSearch (Sequential processing for memory safety)...")
grid_lr = GridSearchCV(lr_pipeline, param_grid=lr_params, cv=3, scoring='accuracy', verbose=1, n_jobs=1)
grid_lr.fit(X_train, y_train)

# --- 4. EVALUATION FUNCTION (The Senior Version) ---
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_probs = model.predict_proba(X_test)[:, 1] # Needed for ROC Curve
    
    print(f"\n{'='*20} {name} Results {'='*20}")
    print("Best Parameters:", model.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    # Visual 1: Confusion Matrix
    plt.figure(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

    # Visual 2: ROC Curve (The Math Proof)
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.savefig(f"{name.lower().replace(' ', '_')}_roc.png")
    plt.show()

    # Visual 3: Feature Importance (Only for Logistic Regression)
    if name == "Logistic Regression":
        print("Extracting Top Indicators...")
        feature_names = model.best_estimator_.named_steps['tfidf'].get_feature_names_out()
        coefficients = model.best_estimator_.named_steps['lr'].coef_[0]
        top_features_idx = np.argsort(coefficients)[-20:]
        
        plt.figure(figsize=(10, 6))
        plt.barh([feature_names[i] for i in top_features_idx], coefficients[top_features_idx], color='teal')
        plt.title("Top 20 Keywords Predicting Fake News")
        plt.savefig("features.png")
        plt.show()

# --- 5. RUN EVALUATION ---
evaluate_model("Logistic Regression", grid_lr, X_test, y_test)