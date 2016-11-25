import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import RidgeClassifier, MultiTaskLasso
from sklearn import svm
from sklearn.base import TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

df = pd.read_csv('data\\movie-reviews\\train.tsv', delimiter='\t', header=0)

print df.count()

print df['Sentiment'].describe()
print df['Sentiment'].value_counts()
print df['Sentiment'].value_counts()/df['Sentiment'].count()

pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english', max_df=0.25, ngram_range=(1, 2), use_idf=False)),
    ('clf', LogisticRegression())
])

parameters = {
    'clf__C': (0.1, 1, 10),
}

X, y = df['Phrase'], df['Sentiment'].as_matrix()
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.5)

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy') 
grid_search.fit(X_train, y_train)

print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t%s: %r' % (param_name, best_parameters[param_name])

predictions = grid_search.predict(X_test)

print 'Accuracy:', accuracy_score(y_test, predictions)
print 'Confusion Matrix:', confusion_matrix(y_test, predictions)
print 'Classification Report:', classification_report(y_test, predictions)