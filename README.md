# Instagram Fake Profile Detection

This project leverages machine learning classifiers to detect fake Instagram profiles. We explore, preprocess, and model the data using several algorithms and combine them using a Voting Classifier for enhanced accuracy.

## Project Overview

1. **Data Loading**
2. **Data Exploration**
3. **Data Preprocessing**
4. **Model Building and Evaluation**
5. **Ensemble Learning with Voting Classifier**
6. **Results and Accuracy**

## Data Loading

We start by loading the training and test datasets, which contain various features indicative of fake and real profiles.

```python
import pandas as pd

# Load the datasets
train_df = pd.read_csv('fake_train.csv')
test_df = pd.read_csv('fake_test.csv')

# Display the first few rows of the training data
train_df.head()





# Display basic information about the training data
train_df.info()

# Summary statistics of the training data
train_df.describe()

# Distribution of the target variable
train_df['fake'].value_counts().plot(kind='bar', title='Distribution of Fake Profiles')


from sklearn.preprocessing import StandardScaler

# Define the target column
target_column = 'fake'

# Separating features and labels
X_train = train_df.drop(target_column, axis=1)
y_train = train_df[target_column]
X_test = test_df.drop(target_column, axis=1)
y_test = test_df[target_column]

# Normalizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)




from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42)
}

# Function to train and evaluate a model
def train_and_evaluate_model(model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f'{model_name} Results:')
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(report)
    print('Confusion Matrix:')
    print(conf_matrix)
    print('\n' + '-'*60 + '\n')

# Train and evaluate each classifier
for model_name, model in classifiers.items():
    train_and_evaluate_model(model, model_name)





from sklearn.ensemble import VotingClassifier

# Define the Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('lr', classifiers['Logistic Regression']), 
        ('knn', classifiers['K-Nearest Neighbors']), 
        ('dt', classifiers['Decision Tree']),
        ('rf', classifiers['Random Forest']),
        ('svm', classifiers['SVM'])
    ],
    voting='hard'  # Change to 'soft' for soft voting
)

# Train and evaluate the Voting Classifier
train_and_evaluate_model(voting_clf, 'Voting Classifier')


Voting Classifier Results:
Accuracy: 0.XX
Classification Report:
              precision    recall  f1-score   support

           0       0.XX      0.XX      0.XX       XXX
           1       0.XX      0.XX      0.XX       XXX

    accuracy                           0.XX       XXX
   macro avg       0.XX      0.XX      0.XX       XXX
weighted avg       0.XX      0.XX      0.XX       XXX

Confusion Matrix:
[[XXX  XX]
 [ XX  XXX]]


