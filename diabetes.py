#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import warnings
warnings.filterwarnings("ignore")

# Loading the csv
health_data = pd.read_csv('healthcare-dataset.csv')
health_data


# In[2]:


# droping the id column
health_data.drop(['id'], axis = 1, inplace=True)
health_data.info()


# In[3]:


# viewing the column with missing data
health_data.isnull().sum()


# In[4]:


# replacing the missing values in bmi column with the mean value

health_data['bmi'].fillna(health_data['bmi'].mean(), inplace=True)


# In[5]:


# checking if there is still missing values
health_data.isnull().sum()


# In[6]:



health_data.describe()


# In[7]:


health_data.corr()


# In[8]:


# Data Visualization

stroke_label =health_data['stroke'].value_counts(sort = True).index
stroke_size = health_data['stroke'].value_counts(sort = True)

pie_colors = ["yellow","red"]
explode = (0.05,0) 
 
plt.figure(figsize=(7,7))
plt.pie(stroke_size, explode=explode, labels=stroke_label, colors=pie_colors, autopct='%1.1f%%', shadow=True, startangle=90,)

plt.title('Quantity of stroke in the dataset')
plt.show()


# In[9]:


# BoxPlot

sns.boxplot(x='avg_glucose_level',data=health_data, color='Red')


# In[10]:


sns.boxplot(x='bmi',data=health_data, color = 'Green')


# In[11]:


sns.boxplot(x='age',data=health_data , color = 'Blue')


# In[12]:


# Count plot for the categorical columns

# Compairing patient with stroke with marital status

plt.figure(figsize=(10,5))
stroke_patient = health_data.loc[health_data['stroke']==1]
sns.countplot(data=stroke_patient,x='ever_married', palette="Set2")
plt.title("Stroke vs Ever-Married")


# In[13]:


# Looks like the number of married people tend to have stroke significantly higher than single people !Interesting


# In[14]:


plt.figure(figsize=(10,5))
stroke_patient = health_data.loc[health_data['stroke']==1]
sns.countplot(data=stroke_patient,x='work_type', palette="Set2")
plt.title("Stroke vs Work Type")


# In[15]:


# People in private sector has higher risk of having a stroke


# In[16]:


plt.figure(figsize=(10,5))
stroke_patient = health_data.loc[health_data['stroke']==1]
sns.countplot(data=stroke_patient,x='smoking_status', palette="Set2")
plt.title("Stroke vs Smoking Status")


# In[17]:


# In total it shows that people who never smoked have more stroke but current smoker and formerly smoked are at high risk


# In[18]:


plt.figure(figsize=(10,5))
stroke_patient = health_data.loc[health_data['stroke']==1]
sns.countplot(data=stroke_patient,x='Residence_type', palette="Set2")
plt.title("Stroke vs Residence Type")


# In[19]:


# Now we have a close distribution of rural and urban type of residence. Looks like it does not effect much


# In[20]:


plt.figure(figsize=(10,5))
stroke_patient = health_data.loc[health_data['stroke']==1]
sns.countplot(data=stroke_patient,x='hypertension', palette="Set2")
plt.title("Stroke vs Hypertension")


# In[22]:


# People without hypertension has more risk to have a stroke


# In[23]:


plt.figure(figsize=(10,5))
stroke_patient = health_data.loc[health_data['stroke']==1]
sns.countplot(data=stroke_patient,x='heart_disease', palette="Set2")
plt.title("Stroke vs Heart Disease")


# In[24]:


# People without any previous heart disease has more risk to have a stroke


# In[25]:


fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
health_data.plot(kind="hist", y="age", bins=70, color="b", ax=axes[0][0])
health_data.plot(kind="hist", y="bmi", bins=100, color="r", ax=axes[0][1])
health_data.plot(kind="hist", y="heart_disease", bins=6, color="g", ax=axes[1][0])
health_data.plot(kind="hist", y="avg_glucose_level", bins=100, color="orange", ax=axes[1][1])
plt.show()


# In[26]:


fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
health_data.plot(kind='scatter', x='age', y='avg_glucose_level', alpha=0.5, color='green', ax=axes[0], title="Age vs. avg_glucose_level")
health_data.plot(kind='scatter', x='bmi', y='avg_glucose_level', alpha=0.5, color='red', ax=axes[1], title="bmi vs. avg_glucose_level")
plt.show()


# In[27]:


# Heatmap

plt.figure(figsize=(10,10))
sns.heatmap(health_data.corr(),annot=True);


# In[28]:


# Pairplot

sns.set(style="ticks");
pair_pal = ["#FA5858", "#58D3F7"]

sns.pairplot(health_data, hue="stroke", palette=pair_pal);
plt.title("stroke");


# In[29]:


# Data Preprocessing


# In[30]:


# Label Encoding

health_data['Residence_type'].unique()


# In[31]:


# residence_encode = {'Urbal': 0, 'Rural': 1}
# health_data['Residence_type'] = health_data['Residence_type'].map(residence_encode)

residence_mapping = {'Urban': 0, 'Rural': 1}
health_data['Residence_type'] = health_data['Residence_type'].map(residence_mapping)


# In[32]:


health_data['ever_married'].unique()


# In[33]:


marriage_encode = {'No': 0, 'Yes': 1}
health_data['ever_married'] = health_data['ever_married'].map(marriage_encode)


# In[34]:


health_data.head()


# In[35]:


# One Hot Encoding

health_data['gender'].unique()


# In[36]:


health_data['gender'] = pd.Categorical(health_data['gender'])
health_dummy_gender = pd.get_dummies(health_data['gender'], prefix = 'encoded_gender')
health_dummy_gender


# In[37]:


health_data['smoking_status'].unique()


# In[38]:


health_data['smoking_status'] = pd.Categorical(health_data['smoking_status'])
health_dummy_smoking_status = pd.get_dummies(health_data['smoking_status'], prefix = 'encoded_smoking_status')
health_dummy_smoking_status


# In[39]:


health_data['work_type'].unique()


# In[40]:


health_data['work_type'] = pd.Categorical(health_data['work_type'])
health_dummy_work_type = pd.get_dummies(health_data['work_type'], prefix = 'encoded_work_type')
health_dummy_work_type


# In[41]:


# Removing gender, work_type and smoking_status column and replacing with the One-Hot-Encode


# In[42]:


health_data.drop("gender", axis=1, inplace=True)
health_data.drop("work_type", axis=1, inplace=True)
health_data.drop("smoking_status", axis=1, inplace=True)


# In[43]:


health_data = pd.concat([health_data, health_dummy_gender], axis=1)
health_data = pd.concat([health_data, health_dummy_smoking_status], axis=1)
health_data = pd.concat([health_data, health_dummy_work_type], axis=1)
health_data


# In[44]:


x = health_data.drop(['stroke'], axis=1)
x


# In[45]:


y = health_data['stroke']
y


# In[46]:


# Splitting datasets into Training set and Testing set


# In[47]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3, random_state= 42)

print(f'Total Number of x dataset: {len(x)}')
print(f'Total Number of y dataset: {len(y)}')
print(f'Total Number of x_train dataset: {len(x_train)}')
print(f'Total Number of y_train dataset: {len(y_train)}')
print(f'Total Number of x_test dataset: {len(x_test)}')
print(f'Total Number of y_test dataset: {len(y_test)}')


# In[48]:


# Feature Scaling 

# StandardScaler standardizes a feature by subtracting the mean and then scaling to unit 
# variance. Unit variance means dividing all the values by the standard deviation. 
# StandardScaler results in a distribution with a standard deviation equal to 1.


# In[49]:


from sklearn.preprocessing import StandardScaler 
std_scaler = StandardScaler()
x_train = std_scaler.fit_transform(x_train)
x_test = std_scaler.transform(x_test)


# In[50]:


def plot_cm(cm,title):
    z = cm
    x = ['No stroke', 'stroke']
    y = x
    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in z]

    # set up figure 
    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='deep')

    # add title
    fig.update_layout(title_text='<i><b>Confusion matrix {}</b></i>'.format(title),
                      #xaxis = dict(title='x'),
                      #yaxis = dict(title='x')
                     )

    # add custom xaxis title
    fig.add_annotation({'font':{'color':"black",'size':14},
                            'x':0.5,
                            'y':-0.10,
                            'showarrow':False,
                            'text':"Predicted value",
                            'xref':"paper",
                            'yref':"paper"})
    
    fig.add_annotation({'font':{'color':"black",'size':14},
                            'x':-0.15,
                            'y':0.5,
                            'showarrow':False,
                            'text':"Real value",
                            'textangle':-90,
                            'xref':"paper",
                            'yref':"paper"})


    # adjust margins to make room for yaxis title
    fig.update_layout(margin={'t':50, 'l':20},width=750,height=750)
    


    # add colorbar
    fig['data'][0]['showscale'] = True
    fig.show()


# In[51]:


def hist_score(score):
    models_names = [
        'Logistic Regression',
    'KNearest Neighbor',
    'Decision Tree Classifier',
    'Random Forest Classifier',
    'SVM',
    ]

    plt.rcParams['figure.figsize']=20,8
    sns.set_style('darkgrid')
    ax = sns.barplot(x=models_names, y=score, palette = "inferno", saturation =2.0)
    plt.xlabel('Classifier Models', fontsize = 20 )
    plt.ylabel('Accuracy in percentage', fontsize = 20)
    plt.title('Accuracy of different Classifier Models on test set', fontsize = 20)
    plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
    plt.yticks(fontsize = 12)
    for i in ax.patches:
        width, height = i.get_width(), i.get_height()
        x, y = i.get_xy() 
        ax.annotate(f'{round(height,2)}%', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
    plt.show()


# In[52]:


# Model Selection


# In[53]:


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[54]:


from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, classification_report, roc_curve, plot_roc_curve, auc, precision_recall_curve, plot_precision_recall_curve, average_precision_score

from sklearn.model_selection import cross_val_score


# In[78]:


def feature_on_model(x_train,y_train,x_test,y_test):
    
    import time
    models= [['Logistic Regression ',LogisticRegression()],
            ['KNearest Neighbor ',KNeighborsClassifier()],
            ['Decision Tree Classifier ',DecisionTreeClassifier()],
            ['Random Forest Classifier ',RandomForestClassifier()],
            ['SVM ',SVC(probability=True)],
            ]

    models_score = []
    t =time.time()
    for name,model in models:

        model = model
        model.fit(x_train,y_train)
        model_pred = model.predict(x_test)
        model_prob = model.predict_proba(x_test)[:,1]
        cm_model = confusion_matrix(y_test, model_pred)
        accuracies = cross_val_score(estimator = model, X = x_train, scoring='accuracy', y = y_train, cv = 10)
        models_score.append(accuracy_score(y_test,model.predict(x_test)) * 100)
        elapsed = time.time() - t
        print('---------------------------------------------------------------')
        print('Done and elapsed time is {}seconds'.format(round(elapsed,3)))
        print(name)
        print('')
        print(classification_report(y_test, model_pred))
        print('')
        print(cm_model)
        print('')
        print('Validation Accuracy: ',accuracy_score(y_test,model.predict(x_test)) * 100)
        print('')
        print('K-Fold Mean Accuracy: ',accuracies.mean() * 100)
        print('')
        plot_cm(cm_model,title=name+"model")
        
        # Roc AUC Curve
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model_prob)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        sns.set_theme(style = 'white')
        plt.figure(figsize = (8, 8))
        plt.plot(false_positive_rate,true_positive_rate, color = '#b01717', label = 'AUC = %0.3f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1], linestyle = '--', color = '#174ab0')
        plt.axis('tight')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend()
        plt.show()
        
    
        
    return models_score


# In[79]:


models_score = feature_on_model(x_train,y_train,x_test,y_test)


# In[80]:


hist_score(models_score)


# In[81]:


# Handling Imbalance data using SMOTE 

# SMOTE - Synthetic Minority Oversampling Technique is an oversampling technique where the 
# synthetic samples are generated for the minority class. This algorithm helps to overcome 
# the overfitting problem posed by random oversampling.


# In[82]:


from imblearn.over_sampling import SMOTE


# In[83]:


print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(y_train==0)))

smote_imbalance = SMOTE(random_state=42)
x_train_res, y_train_res = smote_imbalance.fit_resample(x_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(x_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
print("After OverSampling, counts of label '0': {}".format(sum(y_train_res==0)))


# In[84]:


def feature_on_model(x_train_res,y_train_res,x_test,y_test):
    
    import time
    
    models= [['Logistic Regression ',LogisticRegression()],
            ['KNearest Neighbor ',KNeighborsClassifier()],
            ['Decision Tree Classifier ',DecisionTreeClassifier()],
            ['Random Forest Classifier ',RandomForestClassifier()],
            ['SVM ',SVC(probability=True)],
            ]

    models_score = []
    t=time.time()
    for name,model in models:

        model = model
        model.fit(x_train_res,y_train_res)
        model_pred = model.predict(x_test)
        model_prob = model.predict_proba(x_test)[:,1]
        cm_model = confusion_matrix(y_test, model_pred)
        accuracies = cross_val_score(estimator = model, X = x_train_res, scoring='accuracy', y = y_train_res, cv = 10)
        models_score.append(accuracy_score(y_test,model.predict(x_test)) * 100)
        elapsed = time.time() - t
        
        print('---------------------------------------------------------------')
        print('Done and elapsed time is {}seconds'.format(round(elapsed,3)))
        print(name)
        print('')
        print(classification_report(y_test, model_pred))
        print('')
        print(cm_model)
        print('')
        print('Validation Accuracy: ',accuracy_score(y_test,model.predict(x_test)) * 100)
        print('')
        print('K-Fold Mean Accuracy: ',accuracies.mean() * 100)
        print('')
        plot_cm(cm_model,title=name+"model")
        
        # Roc AUC Curve
        false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model_prob)
        roc_auc = auc(false_positive_rate, true_positive_rate)

        sns.set_theme(style = 'white')
        plt.figure(figsize = (8, 8))
        plt.plot(false_positive_rate,true_positive_rate, color = '#b01717', label = 'AUC = %0.3f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1], linestyle = '--', color = '#174ab0')
        plt.axis('tight')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.legend()
        plt.show()
        
    
        
    return models_score


# In[85]:


models_score = feature_on_model(x_train_res,y_train_res,x_test,y_test)


# In[86]:


hist_score(models_score)


# In[87]:


# Tuning the models


# In[88]:


from sklearn.model_selection import GridSearchCV

# The GridSearchCV is a library function that is a member of sklearn's model_selection 
# package. It helps to loop through predefined hyperparameters and fit your estimator 
# (model) on your training set. So, in the end, you can select the best parameters from 
# the listed hyperparameters.


# In[89]:


gs_models = [(LogisticRegression(),[{'C':[0.25,0.5,0.75,1],'random_state':[0]}]), 
               (KNeighborsClassifier(),[{'n_neighbors':[5,7,8,10], 'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']}]), 
               (SVC(),[{'C':[0.25,0.5,0.75,1],'kernel':['linear', 'rbf'],'random_state':[0]}]), 
               (DecisionTreeClassifier(),[{'criterion':['gini','entropy'],'random_state':[0]}]), 
               (RandomForestClassifier(),[{'n_estimators':[100,150,200],'criterion':['gini','entropy'],'random_state':[0]}]), 
    ]


# In[90]:


import time
for n,m in gs_models:
    t =time.time()
    grid_model = GridSearchCV(estimator=n,param_grid = m, scoring = 'accuracy')
    grid_model.fit(x_train_res, y_train_res)
    accuracy = grid_model.best_score_
    param = grid_model.best_params_
    elapsed = time.time() - t
    print('Done and elapsed time is {}seconds'.format(round(elapsed,3)))
    print('{}:\nBest Accuracy : {:.2f}%'.format(n,accuracy*100))
    print('Best Parameters : ',param)
    print('')
    print('----------------')
    print('')


# In[91]:


#  Model after Tuning Hyperparameters


# In[92]:


# Decision Tree

t =time.time()
dt_classifier = DecisionTreeClassifier(criterion= 'entropy', random_state= 0)
dt_classifier.fit(x_train_res, y_train_res)
y_pred = dt_classifier.predict(x_test)
y_prob = dt_classifier.predict_proba(x_test)[:,1]
c_matrix = confusion_matrix(y_test, y_pred)
elapsed = time.time() - t
print('Done and elapsed time is {}seconds'.format(round(elapsed,3)))

print(classification_report(y_test, y_pred))
print(f'ROC AUC score: {roc_auc_score(y_test, y_prob)}')
print('Accuracy Score: ',accuracy_score(y_test, y_pred) * 100)

# # Visualizing Confusion Matrix

plot_cm(c_matrix,title="Decision Tree Classifier")

# Roc AUC Curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)

sns.set_theme(style = 'white')
plt.figure(figsize = (8, 8))
plt.plot(false_positive_rate,true_positive_rate, color = '#b01717', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], linestyle = '--', color = '#174ab0')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()


# In[117]:


# RandomForest

# Fitting RandomForest Model

t=time.time()
rf_classifier = RandomForestClassifier(criterion= 'gini', n_estimators= 200, random_state= 0)
rf_classifier.fit(x_train_res, y_train_res)
y_pred = rf_classifier.predict(x_test)
y_prob = rf_classifier.predict_proba(x_test)[:,1]
c_matrix = confusion_matrix(y_test, y_pred)
elapsed = time.time() - t
print('Done and elapsed time is {}seconds'.format(round(elapsed,3)))

print(classification_report(y_test, y_pred))
print(f'ROC AUC score: {roc_auc_score(y_test, y_prob)}')
print('Accuracy Score: ',accuracy_score(y_test, y_pred) * 100)

# Visualizing Confusion Matrix
plot_cm(c_matrix,title="Random Forest Classifier")

# Roc AUC Curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

sns.set_theme(style = 'white')
plt.figure(figsize = (8, 8))
plt.plot(false_positive_rate,true_positive_rate, color = '#b01717', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], linestyle = '--', color = '#174ab0')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()


# In[118]:


# KNeighborsClassifier

# Tuning KNeighborsClassifier Model

t=time.time()
kn_classifier = KNeighborsClassifier(metric= 'manhattan', n_neighbors = 5)
kn_classifier.fit(x_train_res, y_train_res)
y_pred = kn_classifier.predict(x_test)
y_prob = kn_classifier.predict_proba(x_test)[:,1]
c_matrix = confusion_matrix(y_test, y_pred)
elapsed = time.time() - t
print('Done and elapsed time is {}seconds'.format(round(elapsed,3)))

print(classification_report(y_test, y_pred))
print(f'ROC AUC score: {roc_auc_score(y_test, y_prob)}')
print('Accuracy Score: ',accuracy_score(y_test, y_pred) * 100)

# Visualizing Confusion Matrix
plot_cm(c_matrix,title="K Nearest Neighbors Classifier")

# Roc AUC Curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

sns.set_theme(style = 'white')
plt.figure(figsize = (8, 8))
plt.plot(false_positive_rate,true_positive_rate, color = '#b01717', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], linestyle = '--', color = '#174ab0')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()


# In[119]:


# Support Vector Classifier

# Tuning SVM Model

t=time.time()
sv_classifier = SVC(C = 1, kernel = 'rbf', random_state = 0, probability = True)
sv_classifier.fit(x_train_res, y_train_res)
y_pred = sv_classifier.predict(x_test)
y_prob = sv_classifier.predict_proba(x_test)[:,1]
c_matrix = confusion_matrix(y_test, y_pred)
elapsed = time.time() - t
print('Done and elapsed time is {}seconds'.format(round(elapsed,3)))

print(classification_report(y_test, y_pred))
print(f'ROC AUC score: {roc_auc_score(y_test, y_prob)}')
print('Accuracy Score: ',accuracy_score(y_test, y_pred) * 100)

# Visualizing Confusion Matrix
plot_cm(c_matrix,title="Support Vector Classifier")

# Roc AUC Curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

sns.set_theme(style = 'white')
plt.figure(figsize = (8, 8))
plt.plot(false_positive_rate,true_positive_rate, color = '#b01717', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], linestyle = '--', color = '#174ab0')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()


# In[120]:


# Logistic Regression

# Tuning Logistic Regression Model

t=time.time()
lr_classifier = LogisticRegression(C = 0.25, random_state = 0)
lr_classifier.fit(x_train_res, y_train_res)
y_pred = lr_classifier.predict(x_test)
y_prob = lr_classifier.predict_proba(x_test)[:,1]
c_matrix = confusion_matrix(y_test, y_pred)
elapsed = time.time() - t
print('Done and elapsed time is {}seconds'.format(round(elapsed,3)))

print(classification_report(y_test, y_pred))
print(f'ROC AUC score: {roc_auc_score(y_test, y_prob)}')
print('Accuracy Score: ',accuracy_score(y_test, y_pred) * 100)

# Visualizing Confusion Matrix
plot_cm(c_matrix,title="Logistic Regression")

# Roc AUC Curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(false_positive_rate, true_positive_rate)

sns.set_theme(style = 'white')
plt.figure(figsize = (8, 8))
plt.plot(false_positive_rate,true_positive_rate, color = '#b01717', label = 'AUC = %0.3f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1], linestyle = '--', color = '#174ab0')
plt.axis('tight')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend()
plt.show()


# In[95]:


# Importing keras

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.regularizers import l2


# In[96]:


# Modeling ANN

def ann_classifier():
    ann_model = tf.keras.models.Sequential()
    ann_model.add(tf.keras.layers.Dense(units= 8, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
    ann_model.add(tf.keras.layers.Dense(units= 8, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
    tf.keras.layers.Dropout(0.6)
    ann_model.add(tf.keras.layers.Dense(units= 1, activation='sigmoid'))
    ann_model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])
    return ann_model


# In[97]:


# # Assigning values into KerasClassifier

# ann_model = KerasClassifier(build_fn = ann_classifier, batch_size = 32, epochs = 50)

# Passing values to KerasClassifier 
ann_model = KerasClassifier(build_fn = ann_classifier, batch_size = 32, epochs = 50)


# In[98]:


# Evaluating the ANN using cross validation
t=time.time()

accuracy = cross_val_score(estimator = ann_model, X = x_train_res, y = y_train_res, cv = 10)
elapsed = time.time() - t
print('Done and elapsed time is {}seconds'.format(round(elapsed,3)))


# In[99]:


mean = accuracy.mean()
std_deviation = accuracy.std()
print("Accuracy: {:.2f} %".format(mean*100))
print("Standard Deviation: {:.2f} %".format(std_deviation*100))


# In[126]:


# Tuning the ANN

# Builing the function
def ann_classifier(optimizer = 'adam'):
    ann_model = tf.keras.models.Sequential()
    ann_model.add(tf.keras.layers.Dense(units= 8, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
    ann_model.add(tf.keras.layers.Dense(units= 8, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
    tf.keras.layers.Dropout(0.6)
    ann_model.add(tf.keras.layers.Dense(units= 1, activation='sigmoid'))
    ann_model.compile(optimizer= optimizer, loss= 'binary_crossentropy', metrics= ['accuracy'])
    return ann_model


# In[127]:


ann_model = KerasClassifier(build_fn = ann_classifier, batch_size = 32, epochs = 50)


# In[128]:


t=time.time()
params = {'batch_size': [25, 32],
             'epochs': [50, 100, 150],
             'optimizer': ['adam', 'rmsprop']}

gs = GridSearchCV(estimator = ann_model, param_grid = params, scoring = 'accuracy', cv = 10, n_jobs = -1)

gs.fit(x_train_res, y_train_res)
elapsed = time.time() - t
print('Done and elapsed time is {}seconds'.format(round(elapsed,3)))


# In[106]:


accuracy = gs.best_score_
parameters = gs.best_params_


# In[107]:


print("Best Accuracy: {:.2f} %".format(accuracy*100))
print("Best Parameters:", parameters)


# In[108]:


# Ann Model after tuning

ann_model = tf.keras.models.Sequential()
ann_model.add(tf.keras.layers.Dense(units= 32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
ann_model.add(tf.keras.layers.Dense(units= 32, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01), activation='relu'))
tf.keras.layers.Dropout(0.6)
ann_model.add(tf.keras.layers.Dense(units= 1, activation='sigmoid'))
ann_model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= ['accuracy'])


# In[122]:


t=time.time()
ann_model_history = ann_model.fit(x_train_res, y_train_res, batch_size= 25, epochs= 150, validation_split= 0.2)
elapsed = time.time() - t
print('Done and elapsed time is {}seconds'.format(round(elapsed,3)))


# In[130]:


mean = accuracy.mean()
std_deviation = accuracy.std()
print("Accuracy: {:.2f} %".format(mean*100))
print("Standard Deviation: {:.2f} %".format(std_deviation*100))


# In[110]:


# Loss Graph

loss_train = ann_model_history.history['loss']
loss_val = ann_model_history.history['val_loss']
epochs = range(1,151)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# In[111]:


# Accuracy Graph
loss_train = ann_model_history.history['accuracy']
loss_val = ann_model_history.history['val_accuracy']
epochs = range(1,151)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[112]:


# Conclusion 
# Therefore, after the multiple visualizations of our and going through all the 
# performance of the models. I tune the hyperparameters with the help of GridSearch to 
# get models. After that, I came to conclusion that RandomForestClassifier is best model 
# for this dataset.


# In[ ]:




