# -*- coding: utf-8 -*-
"""
Linda Zhang
Class: CS 677
Date:4/2/2022
Homework #3
"""

#Loading from packages
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
pd.options.mode.chained_assignment = None

"""
Homework Problem #1.1
Description of Problem:Created a new column for color and made it
green if it is 0 and red if it is 1
"""
df=pd.read_csv('data_banknote_authentication.csv')
df['color']=df['class'].apply(lambda x:'green' if x<1 else 'red')
green_notes=df[df['class']==0]
red_notes=df[df['class']==1]
print('Question 1.1_____________________________________')
print(df.head())

"""
Homework Problem #1.2
Description of Problem:Create a dataframe with calculations of mean and standard
deviation for f1 to f4. 
"""
def graph_calc(data):
    master_list=[]
    for i in [0,1,2,3]:
        m=round(data.mean()[i],3)
        sd=round(data.std()[i],3)
        master_list.append(m)
        master_list.append(sd)
    return master_list

sd_and_mean_table = pd.DataFrame(
    [
     graph_calc(green_notes),
     graph_calc(red_notes),
     graph_calc(df)
     ],
    index=[0,1,'all'],
    columns=['u(f1)','sd(f1)','u(f2)','sd(f2)','u(f3)','sd(f3)','u(f4)','sd(f4)'],
)
print('Question 1.2_____________________________________')
print(sd_and_mean_table)

"""
Homework Problem #2.1
Description of Problem:Splitting the data set for X train and Y train parts.
Saving the graph into a pdf.
"""
train, test=train_test_split(df, test_size=0.5, random_state=1)
train_green=train[train['color']=='green']
train_red=train[train['color']=='red']
train_g=sns.pairplot(train_green)
train_g.savefig('Good_Bills.pdf')
train_b=sns.pairplot(train_red)
train_b.savefig('Bad_Bills.pdf')


"""
Homework Problem #2.2
Description of Problem:Came up with three comparisons I thought would be a
good classifier for my data. I created a function called guess
that would take in the f1, f2 and f3 value and if it fits my
classfier it would return it as a real bill. The results of the
guess would be added into a new column called guess
"""
def guess(data):
    data['guess']=0
    for i in data.index:
        if (data['f1'][i]>-4) and (data['f2'][i]>-5) and (data['f3'][i]>-5):
            data['guess'][i]=0

        else:
            data['guess'][i]=1
    return data['guess']

"""
Homework Problem #2.3
Description of Problem:Applied it to the dataset
"""
test['guess']=guess(test)
print('Question 2.3_____________________________________')
print(test.head())

"""
Homework Problem #2.4
Description of Problem:Created a function that compared the class lablels with
the true labels. 
"""
def calculation(data):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in data.index:
        if data['class'][i]==0:
            if data['guess'][i]==0:
                tp +=1
            else:
                fp +=1
        else:
            if data['guess'][i]==0:
                fn +=1
            else:
                tn +=1
    tpr=round(tp/(tp+fn),2)
    tnr = round(tn/(tn+fp),2)
    accuracy=round((tp+tn)/len(data)*100,2)
    table=[tp,fp,tn,fn,accuracy,tpr,tnr]
    return table


"""
Homework Problem #2.5
Description of Problem:Put the data into a dataframe
"""
accuracy_table_one = pd.DataFrame(
     [calculation(test)],
    columns=['tp','fp','tn','fn','accuracy','tpr','tnr']
)
print('Question 2.5_____________________________________')
print(accuracy_table_one)


"""
Homework Problem #3.1
Description of Problem:Set up the data to be used in a k-NN classifier.
Created a for loop that will run each k-value into the k-NN classfier. 
I then calculated the accuracy of each k-value
"""
#getting the values from x test and x train for the four features
x_test=test[['f1','f2','f3','f4']].values
x_train=train[['f1','f2','f3','f4']].values


scaler = StandardScaler()
scaler.fit(x_train)

x_train= scaler.transform(x_train)
x_test= scaler.transform(x_test)

#getting the values from y test and y train for the four features
y_test=test['class'].values
y_train=train['class'].values

#created an empty list called accurate
accurate=[]
# let k run for (3,5,6,9,11)
print('Question 3.1_____________________________________')
for k in range(3,13,2):
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train,y_train)
    y_predictor=knn.predict(x_test)
    accuracy=metrics.accuracy_score(y_predictor,y_test)
    accurate.append(accuracy)
    print('k=' + str(k) + ', Accuracy:' + str(round(accuracy,4)))


"""
Homework Problem #3.2
Description of Problem:Graph the different k-values
"""
plt.clf()
plt.plot(range(3,13,2),accurate,marker='o')
plt.show()


"""
Homework Problem #3.3
Description of Problem:I used one of the highest k-value accuracy and ran it
through a k-NN classifier. I then compared the highest k-value to x test to
look at the data. I also created a dataframe to put the data in.
"""
knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
new_y_predictor=knn.predict(x_test)
n_tp = 0
n_fp = 0
n_tn = 0
n_fn = 0

for i in range(len(y_predictor)):
    if y_predictor[i]==0:
        if y_test[i]==0:
            n_tp +=1
        else:
            n_fp +=1
    else:
        if y_test[i]==1:
            n_fn +=1
        else:
            n_tn +=1
    i+=1
n_tpr = round(n_tp / (n_tp + n_fn),2)
n_tnr = round(n_tn / (n_tn + n_fp),2)

accurate_table = pd.DataFrame(
    {
        'tp': [n_tp],
        'fp': [n_fp],
        'tn': [n_tn],
        'fn': [n_fn],
        'accuracy': max(accurate),
        'tpr': [n_tpr],
        'tnr': [n_tnr]
    }
)
print('Question 3.3_____________________________________')
print(accurate_table)


"""
Homework Problem #3.5
Description of Problem: We used our BUID to replace the features
to see if the bill would be real or fake. 
"""
print('Question 3.5_____________________________________')
#last 4 digit of buid is 1222
billx=pd.DataFrame([[1,2,2,2]],columns=["f1", "f2", "f3", "f4"])
guess(billx)
print('Using my own predictor my bill is real')


knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(x_train,y_train)
new_new_y_predictor=knn.predict([[1,2,2,2]])
print('Using Knn my bill is real because it returned a value of '+str(new_new_y_predictor))


"""
Homework Problem #4.1
Description of Problem: I created a function that sets up the
feature selection. In the function that I created I made a new x 
test value that would select the columns that were givenES and use
that to make the new prediction. I also used k=7 since it is 
k* and the most optiminal choice.
"""
def feature_selection(columns):
    scaler = StandardScaler()
    new_x_test=test[columns]
    scaler.fit(new_x_test)
    
    new_y_test=(test['class'].values)

    feature_knn=KNeighborsClassifier(n_neighbors=7)
    feature_knn.fit(new_x_test,new_y_test)
    new_feautre_pred=feature_knn.predict(new_x_test)
    accuracy=round(metrics.accuracy_score(new_feautre_pred,new_y_test),3)
    return accuracy
print('Question 4.1_____________________________________')
print(feature_selection(['f2','f3','f4']))
print(feature_selection(['f1','f3','f4']))
print(feature_selection(['f1','f2','f4']))
print(feature_selection(['f1','f2','f3']))


"""
Homework Problem #5.1
Description of Problem: Created logistion regression to test x test
"""
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(x_train,y_train)
logistic_prediction=log_reg_classifier.predict(x_test)


"""
Homework Problem #5.1
Description of Problem: Created a dataframe to summrize the findings
"""
l_tp = 0
l_fp = 0
l_tn = 0
l_fn = 0
for i in range(len(logistic_prediction)):
    if logistic_prediction[i]==0:
        if y_test[i]==0:
            l_tp +=1
        else:
            l_fp +=1
    else:
        if y_test[i]==1:
            l_fn +=1
        else:
            l_tn +=1
    i+=1
l_tpr = round(l_tp / (l_tp + l_fn),2)
l_tnr = round(l_tn / (n_tn + l_fp),2)
log_accuracy=round(metrics.accuracy_score(logistic_prediction,y_test),3)
log_accurate_table = pd.DataFrame(
    {
        "tp": [l_tp],
        "fp": [l_fp],
        "tn": [l_tn],
        "fn": [l_fn],
        "accuracy": [log_accuracy],
        "tpr": [l_tpr],
        "tnr": [l_tnr]
    }
)
print('Question 5.2_____________________________________')
print(log_accurate_table )

"""
Homework Problem #5.5
Description of Problem:Took in my buid to guess if the bill is real or fake
"""
log_reg_classifier = LogisticRegression()
log_reg_classifier.fit(x_train,y_train)
buid_logistic_prediction=log_reg_classifier.predict([[1,2,2,2]])
print('Question 5.5_____________________________________')
print(buid_logistic_prediction)


"""
Homework Problem #5.5
Description of Problem:Using logistic regression but added in removing features
to see which if the dataset would improve in accuract removing one or the other.
"""
def log_feature_selection(columns):
    scaler = StandardScaler()
    new_log_x_test=test[columns]
    scaler.fit(new_log_x_test)
    
    new_log_y_test=(test['class'].values)
    
    log_reg_classifier = LogisticRegression()
    log_reg_classifier.fit(new_log_x_test,new_log_y_test)
    new_logistic_prediction=log_reg_classifier.predict(new_log_x_test)
    accuracy=round(metrics.accuracy_score(new_logistic_prediction,new_log_y_test),3)
    return accuracy
print('Question 6.1_____________________________________')
print(log_feature_selection(['f2','f3','f4']))
print(log_feature_selection(['f1','f3','f4']))
print(log_feature_selection(['f1','f2','f4']))
print(log_feature_selection(['f1','f2','f3']))



