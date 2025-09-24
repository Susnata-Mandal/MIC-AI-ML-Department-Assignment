import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Step 1: Load the IMDB dataset
dataset = pd.read_csv('IMDB Dataset.csv')

# Step 2: Clean the data (remove missing reviews/sentiments)
dataset_clean = dataset.dropna()

# Step 3: Convert sentiment labels to integer values (positive -> 1, negative -> 0)
dataset_clean['sentiment'] = dataset_clean['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Step 4: Separate reviews and labels
reviews = dataset_clean['review']
sentiments = dataset_clean['sentiment']

# Step 5: Transform reviews to feature vectors using Bag of Words
count_vec = CountVectorizer()
review_vectors = count_vec.fit_transform(reviews)

# Step 6: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    review_vectors, sentiments, test_size=0.2, random_state=42
)

# Step 7: Train and evaluate Logistic Regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
acc_logreg = accuracy_score(y_test, pred_logreg)*100  # Now a percentage
print(f"[Logistic Regression] Accuracy: {acc_logreg:.2f}%")

# Step 8: Train and evaluate Multinomial Naive Bayes
nbayes = MultinomialNB()
nbayes.fit(X_train, y_train)
pred_nbayes = nbayes.predict(X_test)
acc_nbayes = accuracy_score(y_test, pred_nbayes)*100  # Now a percentage
print(f"[Naive Bayes] Accuracy: {acc_nbayes:.2f}%")

# Step 9: Show a few predictions for inspection
for i in range(3):
    print(f"\nReview sample {i+1}:\n{reviews.iloc[i][:100]}...")
    print(f"True label: {y_test.iloc[i]}, LR Pred: {pred_logreg[i]}, NB Pred: {pred_nbayes[i]}")

# Step 10: Plot model accuracy comparison
plt.figure(figsize=(6, 4))
models = ['Logistic Regression', 'Naive Bayes']
accuracies = [acc_logreg, acc_nbayes]
bars = plt.bar(models, accuracies, color=['steelblue', 'orange'])
plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')

# Show actual accuracy values on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f"{height:.2f}%", ha='center', va='bottom')

plt.tight_layout()
plt.show()
