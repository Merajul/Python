import pandas as pd
import numpy as np
from sklearn import metrics


food = pd.read_csv('FoodTypeDataset.csv')


feature = food.iloc[:,0:8]
feature = np.array(feature)


target = food.iloc[:,[8]]
target = np.array(target)


from sklearn.preprocessing import StandardScaler
scalerX =StandardScaler().fit(feature)
feature= scalerX.transform(feature)

def getScore(y_test,y_pred,y_train_pred):
    
  
    prec=metrics.precision_score(y_test, y_pred , average='weighted')
    
   
    rec=metrics.recall_score(y_test, y_pred , average='weighted')
    
  
    f1=metrics.f1_score(y_pred, y_test, average='weighted')

  
    acc_train = metrics.accuracy_score(y_train, y_train_pred)

    
    acc_test =metrics.accuracy_score(y_pred, y_test)
    
    return prec,rec,f1,acc_train,acc_test
    

prec_sum1=rec_sum1=f11_sum=sum_acc_train1=sum_acc_test1=0;
prec_sum2=rec_sum2=f12_sum=sum_acc_train2=sum_acc_test2=0;
prec_sum3=rec_sum3=f13_sum=sum_acc_train3=sum_acc_test3=0;
prec_sum4=rec_sum4=f14_sum=sum_acc_train4=sum_acc_test4=0;
prec_sum5=rec_sum5=f15_sum=sum_acc_train5=sum_acc_test5=0;


from sklearn.model_selection import KFold
cv=5
kf = KFold(n_splits=cv,shuffle=True)

for train_index, test_index in kf.split(feature,target):
   X_train, X_test = feature[train_index], feature[test_index]
   y_train, y_test = target[train_index], target[test_index]
   

   from sklearn import linear_model
   clf1 = linear_model.SGDClassifier()
   clf1.fit(X_train, y_train)
       
   y_train_pred1 = clf1.predict(X_train)
   y_pred1 = clf1.predict(X_test)
   
   
   prec1,rec1,f11,acc_train1,acc_test1 = getScore(y_test,y_pred1,y_train_pred1)
   
   
   prec_sum1=prec_sum1+prec1
   rec_sum1=rec_sum1+rec1
   f11_sum=f11_sum+f11
   sum_acc_train1=sum_acc_train1+acc_train1
   sum_acc_test1=sum_acc_test1+acc_test1
   
   
   
   
   
   from sklearn.tree import DecisionTreeClassifier
   clf2 = DecisionTreeClassifier()
   clf2.fit(X_train, y_train)
       
   y_train_pred2 = clf2.predict(X_train)
   y_pred2 = clf2.predict(X_test)
   
   
   prec2,rec2,f12,acc_train2,acc_test2 = getScore(y_test,y_pred2,y_train_pred2)
   
   
   prec_sum2=prec_sum2+prec2
   rec_sum2=rec_sum2+rec2
   f12_sum=f12_sum+f12
   sum_acc_train2=sum_acc_train2+acc_train2
   sum_acc_test2=sum_acc_test2+acc_test2


   
   from sklearn.ensemble import RandomForestClassifier
   clf3 = RandomForestClassifier(n_estimators=200)
   clf3.fit(X_train, y_train)
       
   y_train_pred3 = clf3.predict(X_train)
   y_pred3 = clf3.predict(X_test)
   
   
   prec3,rec3,f13,acc_train3,acc_test3 = getScore(y_test,y_pred3,y_train_pred3)
   
   
   prec_sum3=prec_sum3+prec3
   rec_sum3=rec_sum3+rec3
   f13_sum=f13_sum+f13
   sum_acc_train3=sum_acc_train3+acc_train3
   sum_acc_test3=sum_acc_test3+acc_test3
   
   
 
   
 
   from sklearn.ensemble import AdaBoostClassifier
   clf4 = AdaBoostClassifier()
   clf4.fit(X_train, y_train)
       
   y_train_pred4 = clf4.predict(X_train)
   y_pred4 = clf4.predict(X_test)
   
   
   
   prec4,rec4,f14,acc_train4,acc_test4 = getScore(y_test,y_pred4,y_train_pred4)
   
  
   prec_sum4=prec_sum4+prec4
   rec_sum4=rec_sum4+rec4
   f14_sum=f14_sum+f14
   sum_acc_train4=sum_acc_train4+acc_train4
   sum_acc_test4=sum_acc_test4+acc_test4
   

   
   
   from sklearn import svm
   clf5 = svm.SVC()
   clf5.fit(X_train, y_train)
       
   y_train_pred5 = clf5.predict(X_train)
   y_pred5 = clf5.predict(X_test)
   
   
   
   prec5,rec5,f15,acc_train5,acc_test5 = getScore(y_test,y_pred5,y_train_pred5)
   
   
   prec_sum5=prec_sum5+prec5
   rec_sum5=rec_sum5+rec5
   f15_sum=f15_sum+f15
   sum_acc_train5=sum_acc_train5+acc_train5
   sum_acc_test5=sum_acc_test5+acc_test5
   
 
   
print("-------------------------------------------------------------------")
print("Score for Model 1 : SGDClassifier\n")
print("Average Precision= ",prec_sum1/cv)
print("Average Recall= ",rec_sum1/cv)
print("Average F1_score= ",f11_sum/cv)
print("Average accuracy score for training= ",sum_acc_train1/cv)
print("Average accuracy score for testing= ",sum_acc_test1/cv)
print()


print("-----------------------------------------------------------------------")
print("Score for Model 2 : Decision Tree CLassifier\n")
print("Average Precision= ",prec_sum2/cv)
print("Average Recall= ",rec_sum2/cv)
print("Average F1_score= ",f12_sum/cv)
print("Average accuracy score for training= ",sum_acc_train2/cv)
print("Average accuracy score for testing= ",sum_acc_test2/cv)
print()


print("------------------------------------------------------------------------")
print("Score for Model 3 : Random Forest CLassifier\n")
print("Average Precision= ",prec_sum3/cv)
print("Average Recall= ",rec_sum3/cv)
print("Average F1_score= ",f13_sum/cv)
print("Average accuracy score for training= ",sum_acc_train3/cv)
print("Average accuracy score for testing= ",sum_acc_test3/cv)
print()


print("--------------------------------------------------------------------------")
print("Score for Model 4 : AdaBoost CLassifier\n")
print("Average Precision= ",prec_sum4/cv)
print("Average Recall= ",rec_sum4/cv)
print("Average F1_score= ",f14_sum/cv)
print("Average accuracy score for training= ",sum_acc_train4/cv)
print("Average accuracy score for testing= ",sum_acc_test4/cv)
print()


print("------------------------------------------------------------------------")
print("Score for Model 5 : Support Vector Machine (svm)\n")
print("Average Precision= ",prec_sum5/cv)
print("Average Recall= ",rec_sum5/cv)
print("Average F1_score= ",f15_sum/cv)
print("Average accuracy score for training= ",sum_acc_train5/cv)
print("Average accuracy score for testing= ",sum_acc_test5/cv)
print()


print("------------------------------------------------------------------------")




   


   
   
   
   

