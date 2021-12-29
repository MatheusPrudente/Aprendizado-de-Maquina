from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.impute import SimpleImputer
import numpy as np
import arff
import warnings
warnings.filterwarnings("ignore")

arff_file ='Caminho/dataset_50_tic-tac-toe.arff'

f = open(arff_file, 'r')
full_data = arff.load(f, encode_nominal=True)
f.close()
dataset = np.array(full_data['data'])

num_features = 9

X = dataset[:,0:num_features-1]
y = dataset[:,num_features].astype('int')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print("\n------ SVM Results ------\n")

clf = SVC(C=13.0,kernel='rbf',gamma='auto')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f" % accuracy)

macro_precision = precision_score(y_test, y_pred, average='macro')
print("Macro Precision: %.2f" % macro_precision)
weighted_precision = precision_score(y_test, y_pred, average='weighted')
print("Weighted Precision: %.2f" % weighted_precision)

macro_recall = recall_score(y_test, y_pred, average='macro')
print("Macro Recall: %.2f" % macro_recall)
weighted_recall = recall_score(y_test, y_pred, average='weighted')
print("Weighted Recall: %.2f" % weighted_recall)

macro_fscore = f1_score(y_test, y_pred, average='macro')
print("Macro F-score: %.2f" % macro_fscore)
weighted_fscore = f1_score(y_test, y_pred, average='weighted')
print("Weighted F-score: %.2f" % weighted_fscore)