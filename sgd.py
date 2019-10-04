import pandas as pd
import numpy as np
from sklearn import metrics

food = pd.read_csv('FoodTypeDataset.csv')

feature = food.iloc[:, 0:8]
feature = np.array(feature)


target = food.iloc[:, [8]]
target = np.array(target)


from sklearn.preprocessing import StandardScaler

scalerX = StandardScaler().fit(feature)
feature = scalerX.transform(feature)


def getScore(y_test, y_pred, y_train_pred):

    prec = metrics.precision_score(y_test, y_pred, average='weighted')


    rec = metrics.recall_score(y_test, y_pred, average='weighted')

    
    f1 = metrics.f1_score(y_pred, y_test, average='weighted')

   
    acc_train = metrics.accuracy_score(y_train, y_train_pred)

    
    acc_test = metrics.accuracy_score(y_pred, y_test)

    return prec, rec, f1, acc_train, acc_test

prec_sum1 = rec_sum1 = f11_sum = sum_acc_train1 = sum_acc_test1 = 0;
prec_sum2 = rec_sum2 = f12_sum = sum_acc_train2 = sum_acc_test2 = 0;
prec_sum3 = rec_sum3 = f13_sum = sum_acc_train3 = sum_acc_test3 = 0;
prec_sum4 = rec_sum4 = f14_sum = sum_acc_train4 = sum_acc_test4 = 0;
prec_sum5 = rec_sum5 = f15_sum = sum_acc_train5 = sum_acc_test5 = 0;


from sklearn.model_selection import KFold

cv = 5
kf = KFold(n_splits=cv, shuffle=True)


for train_index, test_index in kf.split(feature, target):
    X_train, X_test = feature[train_index], feature[test_index]
    y_train, y_test = target[train_index], target[test_index]

  
    from sklearn import linear_model

    clf1 = linear_model.SGDClassifier()
    clf1.fit(X_train, y_train)

    y_train_pred1 = clf1.predict(X_train)
    y_pred1 = clf1.predict(X_test)


    prec1, rec1, f11, acc_train1, acc_test1 = getScore(y_test, y_pred1, y_train_pred1)

    
    prec_sum1 = prec_sum1 + prec1
    rec_sum1 = rec_sum1 + rec1
    f11_sum = f11_sum + f11
    sum_acc_train1 = sum_acc_train1 + acc_train1
    sum_acc_test1 = sum_acc_test1 + acc_test1

    print("Score for Model 1 : SGDClassifier\n")
    print("Average Precision= ", prec_sum1 / cv)
    print("Average Recall= ", rec_sum1 / cv)
    print("Average F1_score= ", f11_sum / cv)
    print("Average accuracy score for training= ", sum_acc_train1 / cv)
    print("Average accuracy score for testing= ", sum_acc_test1 / cv)
    print()