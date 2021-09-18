# Diabetes-Prediction-using-machine-learning 
# # step1: Import the depndencies

# In[1]:


#import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # step2: data collection and Analysis

# In[2]:


#Load the pima indian datset
df=pd.read_csv(r"C:\Users\HP\Downloads\diabetes.csv")


# # step 3:Exploratory Data Analysis

# ## 3.1) Undertsanding Your Variables

# #### 3.1.1) Head of the dataset

# In[3]:


#Display the first five records of dataset
df.head()


# In[4]:


#Displsy the last five records of dataset
df.tail()


# In[5]:


#Display randomly any numbers of records of dataset
df.sample(5)


# ### 3.1.2 The shape of the dataset 

# In[6]:


#numbers of rows and column
df.shape


# ### 3.1.3 List type of all columns
# 

# In[7]:


#List type of all columns
df.dtypes


# ### 3.1.4) Info of dataset

# In[8]:


#finding out if the  dataset conatain any null values
df.info()


# ### 3.1.5 ) Summary of the datadset

# In[9]:


#Statistical summary
df.describe()


# ## 3.2) Data cleaning

# #### 3.2.1)drop the Duplicates

# In[10]:


#check the shape before the duplicates 
df.shape


# In[11]:


df=df.drop_duplicates()


# In[12]:


#chck the shape after the duplicate
df.shape


# ### 3.2.2)check the null value

# In[14]:


#Count of null  values
#check the missing values in any column
#display number of null values in every column in dataset
df.isnull().sum()


# In[15]:


#there is no null values in datasdet


# In[16]:


df.columns


# #### check the no. of Zero values in given datset

# In[18]:


print("No.of zero values in Glucose",df[df['Glucose']==0].shape[0])


# In[19]:


print("No.of zero values in BloodPressure",df[df['BloodPressure']==0].shape[0])


# In[20]:


print("No.of zero values in SkinThickness",df[df['SkinThickness']==0].shape[0])


# In[21]:


print("No.of zero values in Insulin",df[df['Insulin']==0].shape[0])


# In[22]:


print("No.of zero values in BMI",df[df['BMI']==0].shape[0])


# #### Replace no of zero values with mean of the columns

# In[24]:


df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())
print("No.of zero values in Glucose",df[df['Glucose']==0].shape[0])


# In[25]:


df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())
print("No.of zero values in BloodPressure",df[df['BloodPressure']==0].shape[0])


# In[26]:


df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].mean())
print("No.of zero values in SkinThickness",df[df['SkinThickness']==0].shape[0])


# In[27]:


df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())
print("No.of zero values in Insulin",df[df['Insulin']==0].shape[0])


# In[28]:


df['BMI']=df['BMI'].replace(0,df['BMI'].mean())
print("No.of zero values in BMI",df[df['BMI']==0].shape[0])


# In[29]:


df.describe()


# # 4)data visualization

# ### 4.1)Count plot

# In[32]:


#outcome count plot
import seaborn as sns
sns.countplot(df['Outcome'],label="Count")


# ### 4.2)Histograms

# In[34]:


# Histogram of each feature
df.hist(bins=10,figsize=(10,10))
plt.show()


# ### 4.3)Scatter plot

# In[36]:


#Scatter plot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(df,figsize=(20,20));


# ### 4.4)Pairplot

# In[38]:


#Pairplot
sns.pairplot(data=df,hue="Outcome")
plt.show()


# ### 4.5)Analyzing relationships between variables
# 

# #### correlationa Analysis

# In[41]:


import seaborn as sns
#get correlation of each datset feature in dataset
corrmat=df.corr()
top_corr_features=corrmat.index
plt.figure(figsize=(10,10))
#plot heatmap
g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# # 5)split the dataframe into X & Y

# In[43]:


target_name ="Outcome"
#separate object for target feature
Y=df[target_name]

#separate object for input featur
X=df.drop(target_name,axis=1)


# In[44]:


X.head()


# In[45]:


Y.head()


# # 6)Apply Feature Scalling 

# In[46]:


# apply the standard scalar
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)
SSX=scaler.transform(X)


# # 7) TRAIN TEST SPLIT

# In[47]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(SSX,Y,test_size=0.2,random_state=4)


# In[48]:


X_train.shape,Y_train.shape


# In[49]:


X_test.shape,Y_test.shape


# # 8)Bulid the CLASSIFICATION Algorithms

# ### 8.1)Logistic Regression 

# In[50]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver='liblinear',multi_class='ovr')
lr.fit(X_train,Y_train)


# ### 8.2)KNeighboursClassifier() (KNN)
# 

# In[51]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)


# ### 8.3) Naive_Bayes Classifier

# In[52]:


from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,Y_train)


# ### 8.4) Support Vector Machine (SVM)

# In[53]:


from sklearn.svm import SVC
sv=SVC()
sv.fit(X_train,Y_train)


# ### 8.5) Decision Tree

# In[54]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,Y_train)


# ### 8.6) Random Forest

# In[55]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(criterion='entropy')
rf.fit(X_train,Y_train)


# # 9) Making Prediction 

# ### 9.1) Making Prediction on test by using Logistic Rregression

# In[56]:


X_test.shape


# In[57]:


# Making Prediction on test datset
lr_pred=lr.predict(X_test)


# In[58]:


lr_pred.shape


# ### 9.2) Making Prediction on test by using KNN

# In[59]:


#Making predition on test dataset 
knn_pred=knn.predict(X_test)


# In[60]:


knn_pred.shape


# ### 9.3) Making prediction on test by using Naivie Bayes

# In[61]:


# Making peditions on test dataset
nb_pred=nb.predict(X_test)


# In[62]:


nb_pred.shape


# ### 9.4) Making prediction on test by using SVM

# In[63]:


# Making peditions on test dataset
sv_pred=sv.predict(X_test)


# In[64]:


sv_pred.shape


# ### 9.5) Making prediction on test by using Decision Tree

# In[65]:


# Making peditions on test dataset
dt_pred=dt.predict(X_test)


# In[66]:


dt_pred.shape


# ### 9.6) Making prediction on test by using Random Forest

# In[67]:


# Making peditions on test dataset
rf_pred=rf.predict(X_test)


# In[68]:


rf_pred.shape


# # 10) Model Evaluation

# ### 10.1)Train Score & Test Score 

# In[69]:


#Train Score & Test Score of Logistic  Regression
from sklearn.metrics import accuracy_score
print("Train Accuracy of Logistic Regression",lr.score(X_train,Y_train)*100)
print("Accuracy (Test) score of Logistic Regression",lr.score(X_test,Y_test)*100)
print("Accuracy score of Logistic Regression",accuracy_score(Y_test,lr_pred)*100)


# In[70]:


#Train Score & Test Score of KNN
from sklearn.metrics import accuracy_score
print("Train Accuracy of KNN",knn.score(X_train,Y_train)*100)
print("Accuracy (Test) score of KNN",knn.score(X_test,Y_test)*100)
print("Accuracy score of KNN",accuracy_score(Y_test,knn_pred)*100)


# In[71]:


#Train Score & Test Score of Naivie Bayes
from sklearn.metrics import accuracy_score
print("Train Accuracy of Naivie Bayes",nb.score(X_train,Y_train)*100)
print("Accuracy (Test) score of Naivie Bayes",nb.score(X_test,Y_test)*100)
print("Accuracy score of Naivie Bayes",accuracy_score(Y_test,nb_pred)*100)


# In[72]:


#Train Score & Test Score of SVM
from sklearn.metrics import accuracy_score
print("Train Accuracy of SVM",sv.score(X_train,Y_train)*100)
print("Accuracy (Test) score of SVM",sv.score(X_test,Y_test)*100)
print("Accuracy score of SVM",accuracy_score(Y_test,sv_pred)*100)


# In[73]:


#Train Score & Test Score of Decision Tree
from sklearn.metrics import accuracy_score
print("Train Accuracy of Decision Tree",df.score(X_train,Y_train)*100)
print("Accuracy (Test) score of Decision Tree",dt.score(X_test,Y_test)*100)
print("Accuracy score of Decision Tree",accuracy_score(Y_test,dt_pred)*100)


# In[ ]:


#Train Score & Test Score of Random Forest
from sklearn.metrics import accuracy_score
print("Train Accuracy of Random Forest",rf.score(X_train,Y_train)*100)
print("Accuracy (Test) score of Random Forest",rf.score(X_test,Y_test)*100)
print("Accuracy score of Random Forest",accuracy_score(Y_test,rf_pred)*100)


# ## 10.2) Confusion Martix

# ### 10.2.1)Confusion matrix of "Logistic Regression"

# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix
# confusion martix of logical Regresion
cm=confusion_matrix(Y_test,lr_pred)
cm


# In[ ]:


sns.heatmap(confusion_matrix(Y_test,lr_pred),annot=True,fmt="d")


# In[ ]:


##print("Classification Report of logical Regresion:\n",classification_report(Y_test,lr_pred,digits=4))


# In[ ]:


TN=cm[0,0]
FP=cm[0,1]
FN=cm[1,0]
TP=cm[1,1]


# In[ ]:


TN,FP,FN,TP


# In[ ]:


# Making Confusion martix of logical Regresion 
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
cm=confusion_matrix(Y_test,lr_pred)
print(" TN-True Negative : ",format(cm[0,0]))
print(" FP-False Negative : ",format(cm[0,1]))
print(" FN-False Negative : ",format(cm[1,0]))
print(" TP-True Positive : ",format(cm[1,1]))
print(" Accuracy Rate : ",format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))*100))
print(" Misclassifiaction Rate: ",format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))


# In[ ]:


80.51948051948052+19.480519480519483


# In[ ]:


import matplotlib.pyplot as plt 
plt.clf()
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Wistia)
classNames=['0','1']
plt.title("Confusion Matrix Of Logical Regression ")
plt.ylabel("Actual(true) values")
plt.xlabel("Predicted values")
tick_marks=np.arange(len(classNames))
plt.xticks(tick_marks, classNames,rotation=45)
plt.yticks(tick_marks,classNames)
s=[['TN','FP'],['FN','TP']]
for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()


# In[ ]:


pd.crosstab(Y_test,lr_pred,margins=False)


# In[ ]:


pd.crosstab(Y_test,lr_pred,margins=True)


# In[ ]:


pd.crosstab(Y_test,lr_pred, rownames=['Actual value'], colnames=['predicted vlues'],margins=True)


# ### PRECISION(PPV-Positive Prediction Value)

# In[ ]:


TP,FP


# In[ ]:


Precision=TP/(TP+FP)
Precision


# In[ ]:


33/(33+11)


# In[ ]:


# print precision score

precision_score = TP/float(TP+FP)*100
print("Precision_Score : {0:0.4} ",format(precision_score))


# In[ ]:


from sklearn.metrics import precision_score
print("precision_score is :",precision_score(Y_test,lr_pred)*100)
print("Micro Average precision score is ",precision_score(Y_test,lr_pred,average='micro')*100)
print("Macro Average precision score is ",precision_score(Y_test,lr_pred,average='macro')*100)
print("Weighetd Average precision score is ",precision_score(Y_test,lr_pred,average='weighted')*100)
print("precision score on non weighted score is ",precision_score(Y_test,lr_pred,average=None)*100)


# In[ ]:


print("Classification Report of Logical Regresion:\n",classification_report(Y_test,lr_pred,digits=4))


# ### Recall(True Positive Rate(TPR))

# In[ ]:


recall_score=TP/float(TP+FN)*100
print("Recall_Score",recall_score)


# In[ ]:


TP,FN


# In[ ]:


33/(33+24)


# In[ ]:


from sklearn.metrics import recall_score
print(" Recall Sensitivity score : ",recall_score(Y_test,lr_pred)*100)
print("Micro Average Recall score is : ",recall_score(Y_test,lr_pred,average='micro')*100)
print("Macro Average Recall score is : ",recall_score(Y_test,lr_pred,average='macro')*100)
print("Weighetd Average Recall score is : ",recall_score(Y_test,lr_pred,average='weighted')*100)
print("Recall score on Non weighted score is : ",recall_score(Y_test,lr_pred,average=None)*100)


# ### False Positive rate (FRP)

# In[ ]:


FPR=FP/float(FP+TN)*100
print("False Positive Rate : {0:0.4} ",format(FPR))


# In[ ]:


FP,TN


# In[ ]:


11/(11+86) 


# ### Specificity

# In[ ]:


specificity=TN/float(TN+FP)*100
print("Specificity : {0:0.4f} ",format(specificity))


# ### F1-Score

# In[ ]:


from sklearn.metrics import f1_score
print(" f1_score of macro : ",f1_score(Y_test,lr_pred)*100)
print("Micro Average f1_score is : ",f1_score(Y_test,lr_pred,average='micro')*100)
print("Macro Average f1_score is : ",f1_score(Y_test,lr_pred,average='macro')*100)
print("Weighetd Average f1_score is : ",f1_score(Y_test,lr_pred,average='weighted')*100)
print("f1_score on Non weighted score is : ",f1_score(Y_test,lr_pred,average=None)*100)


# ### Classification Report of Logical Regression 

# In[ ]:


from sklearn.metrics import classification_report
print("Classification Report of Logical Regresion:\n",classification_report(Y_test,lr_pred,digits=4))


# ### ROC Curve & ROC AUC

# In[ ]:


# Area under curve Logistic Regression
auc = roc_auc_score(Y_test,lr_pred)
print("ROC Curve & ROC AUC Logistic Regression is",auc)


# In[ ]:


fpr,tpr,thresholds=roc_curve(Y_test,lr_pred)
plt.plot(fpr,tpr,color='orange',label='ROC')
plt.plot([0,1],[0,1],color='darkblue',linestyle='--',label='ROC curve(area=%0.2f) '%auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate ')
plt.title('Receiver Operating Characteristic (ROC) Curve of Logistic Regression')
plt.legend()
plt.grid()
plt.show()


# ### 10.2.2 Confusion Matrix of "KNN"

# In[ ]:


sns.heatmap(confusion_matrix(Y_test,knn_pred),annot=True,fmt="d")


# In[ ]:


# Making Confusion martix of KNN
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
cm=confusion_matrix(Y_test,knn_pred)
print(" TN-True Negative : ",format(cm[0,0]))
print(" FP-False Negative : ",format(cm[0,1]))
print(" FN-False Negative : ",format(cm[1,0]))
print(" TP-True Positive : ",format(cm[1,1]))
print(" Accuracy Rate : ",format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))*100))
print(" Misclassifiaction Rate: ",format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))


# In[ ]:


72.72727272727273+27.27272727272727


# ### Area under curve of KNN

# In[ ]:


# Area under curve of KNN
auc = roc_auc_score(Y_test,knn_pred)
print("ROC Curve & ROC AUC KNN is",auc)


# In[ ]:


fpr,tpr,thresholds=roc_curve(Y_test,knn_pred)
plt.plot(fpr,tpr,color='orange',label='ROC')
plt.plot([0,1],[0,1],color='darkblue',linestyle='--',label='ROC curve(area=%0.2f) '%auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate ')
plt.title('Receiver Operating Characteristic (ROC) Curve of KNN')
plt.legend()
plt.grid()
plt.show()


# ### 10.2.3) Confusion Matrix of "Naivie Bayes"

# In[ ]:


# Making Confusion martix of Naivie Bayes
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
cm=confusion_matrix(Y_test,nb_pred)
print(" TN-True Negative : ",format(cm[0,0]))
print(" FP-False Negative : ",format(cm[0,1]))
print(" FN-False Negative : ",format(cm[1,0]))
print(" TP-True Positive : ",format(cm[1,1]))
print(" Accuracy Rate : ",format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))*100))
print(" Misclassifiaction Rate: ",format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))


# In[ ]:


74.67532467532467+25.324675324675322


# In[ ]:


sns.heatmap(confusion_matrix(Y_test,nb_pred),annot=True,fmt="d")


# ### Classification Report of Naivie Bayes

# In[ ]:


from sklearn.metrics import classification_report
print("Classification Report of Logical Regresion:\n",classification_report(Y_test,nb_pred,digits=4))


# ### ROC AUC Score of Naivie Bayes

# In[ ]:


# Area under curve Naivie Bayes
auc = roc_auc_score(Y_test,nb_pred)
print("ROC Curve & ROC AUC Naivie Bayes is",auc)


# ### 10.2.4)fusion Matrix of "SVM"

# In[ ]:


sns.heatmap(confusion_matrix(Y_test,sv_pred),annot=True,fmt="d")


# In[ ]:


# Making Confusion martix of SVM
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
cm=confusion_matrix(Y_test,sv_pred)
print(" TN-True Negative : ",format(cm[0,0]))
print(" FP-False Negative : ",format(cm[0,1]))
print(" FN-False Negative : ",format(cm[1,0]))
print(" TP-True Positive : ",format(cm[1,1]))
print(" Accuracy Rate : ",format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))*100))
print(" Misclassifiaction Rate: ",format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))


# In[ ]:


76.62337662337663+23.376623376623375


# ### Classification Report of SVM

# In[ ]:


print("Classification Report of SVM:\n",classification_report(Y_test,sv_pred,digits=4))


# ### ROC AUC Score of SVM

# In[ ]:


from sklearn.metrics import roc_auc_score
auc = round(roc_auc_score(Y_test,sv_pred)*100,2)
print("roc_auc_score SVC is",auc)


# In[ ]:


fpr,tpr,thresholds=roc_curve(Y_test,sv_pred)
plt.plot(fpr,tpr,color='orange',label='ROC')
plt.plot([0,1],[0,1],color='darkblue',linestyle='--',label='ROC curve(area=%0.2f) '%auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate ')
plt.title('Receiver Operating Characteristic (ROC) Curve of KNN')
plt.legend()
plt.grid()
plt.show()


# ### 10.2.5) Confusion Matrix of "Decision Tree"

# In[ ]:


# Making Confusion martix of Decision Tree
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
cm=confusion_matrix(Y_test,dt_pred)
print(" TN-True Negative : ",format(cm[0,0]))
print(" FP-False Negative : ",format(cm[0,1]))
print(" FN-False Negative : ",format(cm[1,0]))
print(" TP-True Positive : ",format(cm[1,1]))
print(" Accuracy Rate : ",format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))*100))
print(" Misclassifiaction Rate: ",format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))


# In[ ]:


72.72727272727273+ 27.27272727272727


# ### Classification Report of Decision Tree

# In[ ]:


print("Classification Report of Decision Tree:\n",classification_report(Y_test,dt_pred,digits=4))


# In[ ]:


#ROC AUC Score of Decision Tree


# In[ ]:


from sklearn.metrics import roc_auc_score
auc = round(roc_auc_score(Y_test,dt_pred)*100,2)
print("roc_auc_score Decision Tree is",auc)


# In[ ]:


fpr,tpr,thresholds=roc_curve(Y_test,dt_pred)
plt.plot(fpr,tpr,color='orange',label='ROC')
plt.plot([0,1],[0,1],color='darkblue',linestyle='--',label='ROC curve(area=%0.2f) '%auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate ')
plt.title('Receiver Operating Characteristic (ROC) Curve of Decision Tree')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


sns.heatmap(confusion_matrix(Y_test,dt_pred),annot=True,fmt="d")


# ### 10.2.6)Confusion Matrix of "Random Forest"

# In[ ]:


# Making Confusion martix of Random Forest
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve
cm=confusion_matrix(Y_test,rf_pred)
print(" TN-True Negative : ",format(cm[0,0]))
print(" FP-False Negative : ",format(cm[0,1]))
print(" FN-False Negative : ",format(cm[1,0]))
print(" TP-True Positive : ",format(cm[1,1]))
print(" Accuracy Rate : ",format(np.divide(np.sum([cm[0,0],cm[1,1]]),np.sum(cm))*100))
print(" Misclassifiaction Rate: ",format(np.divide(np.sum([cm[0,1],cm[1,0]]),np.sum(cm))*100))


# In[ ]:


75.97402597402598+24.025974025974026


# ### Classification Report of Random Forest

# In[ ]:


print("Classification Report of Random Forest:\n",classification_report(Y_test,rf_pred,digits=4))


# In[ ]:


#ROC AUC Score of Random Forest


# In[ ]:


from sklearn.metrics import roc_auc_score
auc = round(roc_auc_score(Y_test,rf_pred)*100,2)
print("roc_auc_score Random Forest  is",auc)


# In[ ]:


fpr,tpr,thresholds=roc_curve(Y_test,rf_pred)
plt.plot(fpr,tpr,color='orange',label='ROC')
plt.plot([0,1],[0,1],color='darkblue',linestyle='--',label='ROC curve(area=%0.2f) '%auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate ')
plt.title('Receiver Operating Characteristic (ROC) Curve of Random Forest')
plt.legend()
plt.grid()
plt.show()


# In[ ]:


sns.heatmap(confusion_matrix(Y_test,rf_pred),annot=True,fmt="d")


# In[ ]:





# In[ ]:




