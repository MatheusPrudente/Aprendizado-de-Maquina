from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.impute import SimpleImputer
import numpy as np
import arff
import warnings
warnings.filterwarnings("ignore")

#ALTERE O NOME/CAMINHO DA BASE DE DADOS AQUI
arff_file ='Caminho/dataset_50_tic-tac-toe.arff'


#ALTERE A QUANTIDADE DE ATRIBUTOS QUE VOCÊ TEM NA BASE AQUI
num_features = 9

f = open(arff_file, 'r')
full_data = arff.load(f, encode_nominal=True)
f.close()
dataset = np.array(full_data['data'])

num_features = 9

X = dataset[:,0:num_features-1]
y = dataset[:,num_features].astype('int')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#CONFIGURE DIFERENTES PARÂMETROS AQUI
clf_svm = SVC(C=13.0,kernel='rbf',gamma='auto',probability = True)
clf_svm.fit(X_train, y_train)
y_proba_svm = clf_svm.predict_proba(X_test)

#CONFIGURE DIFERENTES PARÂMETROS AQUI
clf_nb = GaussianNB()
clf_nb.fit(X_train, y_train)
y_proba_nb = clf_nb.predict_proba(X_test)

#CONFIGURE DIFERENTES PARÂMETROS AQUI
clf_dt = DecisionTreeClassifier(criterion = 'entropy',max_features=2)
clf_dt.fit(X_train, y_train)
y_proba_dt = clf_dt.predict_proba(X_test)

#CONFIGURE DIFERENTES PARÂMETROS AQUI
clf_mlp = MLPClassifier(solver='lbfgs')
clf_mlp.fit(X_train, y_train)
y_proba_mlp = clf_mlp.predict_proba(X_test)


#CONFIGURE DIFERENTES PARÂMETROS AQUI
clf_knn = KNeighborsClassifier(n_neighbors=6)
clf_knn.fit(X_train, y_train)
y_proba_knn = clf_knn.predict_proba(X_test)



def op_sum(x, y):
    return x + y

def op_prod(x, y):
    return x * y

def combinar_probs(class_order, y_proba_1, y_proba_2, op):
    num_classes = len(class_order)
    y_proba_comb = []
    for probs_line_1, probs_line_2 in zip(y_proba_1, y_proba_2):
        y_proba_line = []
        for i in range(num_classes):
            prob_comb = op(probs_line_1[i], probs_line_2[i])
            y_proba_line.append(prob_comb)
        y_proba_comb.append(y_proba_line)
    return y_proba_comb

def idenficar_pred(y_proba_comb, class_order):
    y_pred = []
    for prob_line in y_proba_comb:
        max_prob = max(prob_line)
        class_index = prob_line.index(max_prob)
        pred = class_order[class_index]
        y_pred.append(pred)
    return y_pred

def imprimir_resultados(y_test, y_pred):
    
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

class_order = clf_svm.classes_

#Combinando SVM com NB
y_proba_comb_sum = combinar_probs(class_order, y_proba_svm, y_proba_nb, op_sum)
y_pred_sum = idenficar_pred(y_proba_comb_sum, class_order)
y_proba_comb_prod = combinar_probs(class_order, y_proba_svm, y_proba_nb, op_prod)
y_pred_prod = idenficar_pred(y_proba_comb_prod, class_order)

print("\n------ Results of SVM + NB Late Fusion with Product Rule ------\n")
imprimir_resultados(y_test, y_pred_prod)
print("\n------ Results of SVM + NB Late Fusion with Sum Rule ------\n")
imprimir_resultados(y_test, y_pred_sum)

#Combinando SVM com MLP
y_proba_comb_sum = combinar_probs(class_order, y_proba_svm, y_proba_mlp, op_sum)
y_pred_sum = idenficar_pred(y_proba_comb_sum, class_order)
y_proba_comb_prod = combinar_probs(class_order, y_proba_svm, y_proba_mlp, op_prod)
y_pred_prod = idenficar_pred(y_proba_comb_prod, class_order)

print("\n------ Results of SVM + MLP Late Fusion with Product Rule ------\n")
imprimir_resultados(y_test, y_pred_prod)
print("\n------ Results of SVM + MLP Late Fusion with Sum Rule ------\n")
imprimir_resultados(y_test, y_pred_sum)

#Combinando SVM com DT
y_proba_comb_sum = combinar_probs(class_order, y_proba_svm, y_proba_dt, op_sum)
y_pred_sum = idenficar_pred(y_proba_comb_sum, class_order)
y_proba_comb_prod = combinar_probs(class_order, y_proba_svm, y_proba_dt, op_prod)
y_pred_prod = idenficar_pred(y_proba_comb_prod, class_order)

print("\n------ Results of SVM + DT Late Fusion with Product Rule ------\n")
imprimir_resultados(y_test, y_pred_prod)
print("\n------ Results of SVM + DT Late Fusion with Sum Rule ------\n")
imprimir_resultados(y_test, y_pred_sum)

#Combinando SVM com KNN
y_proba_comb_sum = combinar_probs(class_order, y_proba_svm, y_proba_knn, op_sum)
y_pred_sum = idenficar_pred(y_proba_comb_sum, class_order)
y_proba_comb_prod = combinar_probs(class_order, y_proba_svm, y_proba_knn, op_prod)
y_pred_prod = idenficar_pred(y_proba_comb_prod, class_order)

print("\n------ Results of SVM + KNN Late Fusion with Product Rule ------\n")
imprimir_resultados(y_test, y_pred_prod)
print("\n------ Results of SVM + KNN Late Fusion with Sum Rule ------\n")
imprimir_resultados(y_test, y_pred_sum)

#Combinando DT com MLP
y_proba_comb_sum = combinar_probs(class_order, y_proba_dt, y_proba_mlp, op_sum)
y_pred_sum = idenficar_pred(y_proba_comb_sum, class_order)
y_proba_comb_prod = combinar_probs(class_order, y_proba_dt, y_proba_mlp, op_prod)
y_pred_prod = idenficar_pred(y_proba_comb_prod, class_order)

print("\n------ Results of DT + MLP Late Fusion with Product Rule ------\n")
imprimir_resultados(y_test, y_pred_prod)
print("\n------ Results of DT + MLP Late Fusion with Sum Rule ------\n")
imprimir_resultados(y_test, y_pred_sum)

#Combinando DT com NB
y_proba_comb_sum = combinar_probs(class_order, y_proba_dt, y_proba_nb, op_sum)
y_pred_sum = idenficar_pred(y_proba_comb_sum, class_order)
y_proba_comb_prod = combinar_probs(class_order, y_proba_dt, y_proba_nb, op_prod)
y_pred_prod = idenficar_pred(y_proba_comb_prod, class_order)

print("\n------ Results of DT + NB Late Fusion with Product Rule ------\n")
imprimir_resultados(y_test, y_pred_prod)
print("\n------ Results of DT + NB Late Fusion with Sum Rule ------\n")
imprimir_resultados(y_test, y_pred_sum)


#Combinando DT com KNN
y_proba_comb_sum = combinar_probs(class_order, y_proba_dt, y_proba_knn, op_sum)
y_pred_sum = idenficar_pred(y_proba_comb_sum, class_order)
y_proba_comb_prod = combinar_probs(class_order, y_proba_dt, y_proba_knn, op_prod)
y_pred_prod = idenficar_pred(y_proba_comb_prod, class_order)

print("\n------ Results of DT + KNN Late Fusion with Product Rule ------\n")
imprimir_resultados(y_test, y_pred_prod)
print("\n------ Results of DT + KNN Late Fusion with Sum Rule ------\n")
imprimir_resultados(y_test, y_pred_sum)

#Combinando MLP com NB
y_proba_comb_sum = combinar_probs(class_order, y_proba_mlp, y_proba_nb, op_sum)
y_pred_sum = idenficar_pred(y_proba_comb_sum, class_order)
y_proba_comb_prod = combinar_probs(class_order, y_proba_mlp, y_proba_nb, op_prod)
y_pred_prod = idenficar_pred(y_proba_comb_prod, class_order)

print("\n------ Results of MLP + NB Late Fusion with Product Rule ------\n")
imprimir_resultados(y_test, y_pred_prod)
print("\n------ Results of MLP + NB Late Fusion with Sum Rule ------\n")
imprimir_resultados(y_test, y_pred_sum)

#Combinando MLP com KNN
y_proba_comb_sum = combinar_probs(class_order, y_proba_mlp, y_proba_knn, op_sum)
y_pred_sum = idenficar_pred(y_proba_comb_sum, class_order)
y_proba_comb_prod = combinar_probs(class_order, y_proba_mlp, y_proba_knn, op_prod)
y_pred_prod = idenficar_pred(y_proba_comb_prod, class_order)

print("\n------ Results of MLP + KNN Late Fusion with Product Rule ------\n")
imprimir_resultados(y_test, y_pred_prod)
print("\n------ Results of MLP + KNN Late Fusion with Sum Rule ------\n")
imprimir_resultados(y_test, y_pred_sum)

#Combinando NB com KNN
y_proba_comb_sum = combinar_probs(class_order, y_proba_nb, y_proba_knn, op_sum)
y_pred_sum = idenficar_pred(y_proba_comb_sum, class_order)
y_proba_comb_prod = combinar_probs(class_order, y_proba_nb, y_proba_knn, op_prod)
y_pred_prod = idenficar_pred(y_proba_comb_prod, class_order)

print("\n------ Results of NB + KNN Late Fusion with Product Rule ------\n")
imprimir_resultados(y_test, y_pred_prod)
print("\n------ Results of NB + KNN Late Fusion with Sum Rule ------\n")
imprimir_resultados(y_test, y_pred_sum)



