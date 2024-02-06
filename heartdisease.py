import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('D:\Final yr project\Datasets/heart_disease_health_indicators_BRFSS2015.csv')
df.head()
df.tail()
df.shape
df.columns
df.dtypes
df.describe()
df.info()

categories = []
numerical = []
for i in df.columns:
    if(df[i].dtypes == 'object'):
        categories.append(i)
    else:
        numerical.append(i)


print("Categorical Value: \n", categories)
print("\nContinuous Value: \n", numerical)

#Missing Value Detection

df.isnull()
df.isnull().sum()

#Outlier Detection

# Check for outlier using multiple boxplot

outlier = [ 'BMI', 'GenHlth', 'MentHlth', 'PhysHlth',  'Age', 'Education', 'Income']

for column in outlier:
    if df[column].dtype in ['int64', 'float64']:
        plt.figure()
        df.boxplot(column = [column])


new_df = df.copy()

Limits = pd.DataFrame({
    'Feature Name': [],
    'Upper_Limit': [],
    'Lower_Limit': []})

for feature in outlier:
    # Here we are computing the first and third quartile and interquartile range i.e. IQR to compute upper & lower limit
    q1 = df[feature].quantile(0.25)
    q3 = df[feature].quantile(0.75)
    iqr = q3 - q1

    upper_limit = q3 + (1.5 * iqr)
    lower_limit = q1 - (1.5 * iqr)

    dicts = [{'Feature Name': feature, 'Upper_Limit': upper_limit, 'Lower_Limit': lower_limit}]

    Limits = pd.concat([Limits, pd.DataFrame(dicts)], ignore_index=True, sort=False)


    # Capping i.e. equating the values of the outlier to their upper and lower limit
    new_df.loc[(new_df[feature]>upper_limit), feature] = upper_limit
    new_df.loc[(new_df[feature]<lower_limit), feature] = lower_limit

print(Limits)

for column in outlier:
    if new_df[column].dtype in ['int64', 'float64']:
        plt.figure()
        plt.title("After Outlier Removal")
        new_df.boxplot(column = [column])

#Univariate Analysis

new_df['HeartDiseaseorAttack'].value_counts()

def analysis(feature):
    sns.set_style('darkgrid')
    plt.figure(figsize=(8,6))
    sns.distplot(new_df[feature])
    plt.show()

analysis('Age')

analysis('BMI')

data0=new_df[new_df['HeartDiseaseorAttack']==0.0]

data1=new_df[new_df['HeartDiseaseorAttack']==1.0]

fig,axes=plt.subplots(1,2,figsize=(10,5))

sns.histplot(ax=axes[0],x=data0['Age'])
sns.histplot(ax=axes[1],x=data1["Age"])
axes[0].set_title("no disease")
axes[1].set_title("disease")

print(data1['Sex'].value_counts())

l=list((data1['Sex'].value_counts()))
l

plt.title('Proportion of male and female which are having heart disease')
plt.pie(l,labels=['Male','Female'])


plt.show()

new_df['Sex'] = new_df['Sex'].map({1: 'Masculine', 0: 'Feminine'})
count_gender = new_df['Sex'].value_counts()
print(count_gender)

plt.figure(figsize=(8, 8))
plt.pie(count_gender, labels=count_gender.index, autopct='%1.1f%%', colors=['#fcfdbf', '#fc8961'])
plt.title('Proportion of Men and Women among Respodents')
plt.show()

mapping_age_ranges = {
    1: '18-24',
    2: '25-29',
    3: '30-34',
    4: '35-39',
    5: '40-44',
    6: '45-49',
    7: '50-54',
    8: '55-59',
    9: '60-64',
    10: '65-69',
    11: '70-74',
    12: '75-79',
    13: '80 or above'
}

# Apply the mapping to the age column
new_df['Age'] = new_df['Age'].map(mapping_age_ranges)

age_order = ['18-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 or above']
new_df['Age'] = pd.Categorical(new_df['Age'], categories=age_order, ordered=True)

#Plot bar chart for distribution by age group
plt.figure(figsize=(14, 6))
sns.countplot(x='Age', data=new_df, palette='rocket_r')
#sns.palplot(sns.light_palette("red"))
plt.title('Distribution by Age group among Respodents')
plt.ylabel('Number of Interviewees')
plt.xlabel('Age Group')
plt.xticks(rotation=45, ha='right')  # Rotation labels on the x-axis for better visualization
plt.show()

#Bivariate Analysis

corr = new_df.corr()
corr
plt.figure(figsize=(20,18))
sns.heatmap(corr, xticklabels = corr.columns, yticklabels = corr.columns, annot = True)

#Encoding the categorical data

# encoding categorical values
cat_features = ["Sex", "Age"]
new_df = pd.get_dummies(new_df, columns = cat_features)
print(new_df.columns)

new_df.head()

#Model Building

target = new_df['HeartDiseaseorAttack']
train = new_df.drop('HeartDiseaseorAttack', axis=1)

# Split data into train and test
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=20)

# Apply SVM Model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm

svc = SVC()

svc.fit(x_train, y_train)

y_pred=svc.predict(x_test)
print('Model accuracy : {0:0.3f}'. format(accuracy_score(y_test, y_pred)))
ac_svm = accuracy_score(y_test, y_pred)
print()
print("Confusion Matrix: ")
print()
cm_svm = cm(y_test, y_pred)
cm_svm

#Logistic regression
from sklearn.linear_model import LogisticRegression
lreg = LogisticRegression()
lreg.fit(x_train, y_train)

prediction = lreg.predict(x_test)
print('Model accuracy : {0:0.3f}'. format(accuracy_score(y_test, prediction)))
ac_lr = accuracy_score(y_test, prediction)
print()
print("Confusion Matrix: ")
print()
cm_lr = cm(y_test, prediction)
cm_lr

#Descision tree classifer

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=4, random_state=0)
model = clf.fit(x_train, y_train)
model
predict = clf.predict(x_test)
print('Model accuracy : {0:0.3f}'. format(accuracy_score(y_test, predict)))
ac_dtc = accuracy_score(y_test, predict)
print()
print("Confusion Matrix: ")
print()
cm_dtc = cm(y_test, predict)
cm_dtc

#random forest classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(x_train, y_train)
predicted = rfc.predict(x_test)
print('Model accuracy : {0:0.3f}'. format(accuracy_score(y_test, predicted)))
ac_rfc = accuracy_score(y_test, predicted)
print()
print("Confusion Matrix: ")
print()
cm_rfc = cm(y_test, predicted)
cm_rfc

#Comparison of test accuracy of all Four trained models


print("Support Vector Machine : ",ac_svm)
print("Logistic Regression : ",ac_lr)
print("Decision Tree Classifier : ",ac_dtc)
print("Random Forest Classifier : ", ac_rfc)

#Comparison of all Four confusion matrices of trained models

print("\nSupport Vector Machine confusion matrix: ")
print(cm(y_test, y_pred))
print("\nLogistic Regression confusion matrix: ")
print(cm(y_test, prediction))
print("\nDecision Tree Classifier confusion matrix: ")
print(cm(y_test, predict))
print("\nRandom Forest Classifier confusion matrix: ")
print(cm(y_test, predicted))