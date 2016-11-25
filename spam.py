import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

df = pd.read_csv('data\SMSSpamCollection.txt', delimiter='\t', header=None)

print df.head()

print 'Number of spam messages:', df[df[0] == 'spam'][0].count()
print 'Number of ham messages:', df[df[0] == 'ham'][0].count()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(df[1], df[0])

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

classifier = LogisticRegression()
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

print 'Cross val scores:', cross_val_score(classifier, X_train, y_train)

# Confusion Matrix and Accuracy

confusion_matrix = confusion_matrix(y_test, predictions)

print(confusion_matrix)
print 'Accuracy:', accuracy_score(y_test, predictions)

# Precision, Recall and F1 Score 

precision, recall, fscore, support = score(y_test, predictions)

print 'Precision: ', precision
print 'Recall: ', recall
print 'F1 Score: ', fscore
print 'Support: ', support

# ROC AUC

y = label_binarize(y_test, classes=['ham', 'spam'])
pred_prob = classifier.predict_proba(X_test)

false_positive_rate, recall, thresholds = roc_curve(y, pred_prob[:, 1])
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic')
plt.plot(false_positive_rate, recall, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out')
plt.show()

# Tuning models with grid search

pipeline = Pipeline([
    ('vect', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])

parameters = {
    'vect__max_df': (0.25, 0.5, 0.75),
    'vect__stop_words': ('english', None),
    'vect__max_features': (2500, 5000, 10000, None),
    'vect__ngram_range': ((1, 1), (1, 2)),
    'vect__use_idf': (True, False),
    'vect__norm': ('l1', 'l2'),
    'clf__penalty': ('l1', 'l2'),
    'clf__C': (0.01, 0.1, 1, 10),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy', cv=3)

X_train, X_test, y_train, y_test = train_test_split(df[1], df[0])
grid_search.fit(X_train, y_train)
print 'Best score: %0.3f' % grid_search.best_score_
print 'Best parameters set:'
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print '\t%s: %r' % (param_name, best_parameters[param_name])
predictions = grid_search.predict(X_test)

precision, recall, fscore, support = score(y_test, predictions)
print 'Accuracy:', accuracy_score(y_test, predictions)
print 'Precision:', precision
print 'Recall:', recall