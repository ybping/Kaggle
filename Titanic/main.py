
# coding: utf-8

# In[172]:

# https://www.kaggle.com/ybping/titanic/titanics/editnb


# In[99]:

import pandas as pd
import numpy as np


# In[131]:

train_data = pd.read_csv('./input/train.csv', dtype={"Age": np.float64})


# In[132]:

test_data = pd.read_csv('./input/test.csv', dtype={"Age": np.float64})


# In[133]:

train_data = train_data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
test_data = test_data.drop(['Name', 'Ticket'], axis=1)


# In[134]:

x_train = train_data.drop("Survived", axis=1)
y_train = train_data['Survived']
x_test = test_data.drop('PassengerId',axis=1)


# In[135]:

x_train.drop(['Sex', 'Embarked', 'Cabin'], inplace=True, axis=1)
x_test.drop(['Sex', 'Embarked', 'Cabin'], inplace=True, axis=1)


# In[152]:

age_x_train_mean = x_train['Age'].mean()
age_x_train_std = x_train['Age'].std()
count_age_train_nan = x_train['Age'].isnull().sum()

age_x_test_mean = x_test['Age'].mean()
age_x_test_std = x_test['Age'].std()
count_age_test_nan = x_test['Age'].isnull().sum()

import numpy as np
age_x_train_rand = np.random.randint(age_x_train_mean - age_x_train_std, age_x_train_mean + age_x_train_std, count_age_train_nan)
age_x_test_rand = np.random.randint(age_x_test_mean - age_x_test_std, age_x_test_mean + age_x_test_std, count_age_test_nan)


x_train['Age'][x_train['Age'].isnull()] = age_x_train_rand
x_test['Age'][x_test['Age'].isnull()] = age_x_test_rand


# In[165]:

x_test['Fare'].fillna(x_test['Fare'].median(), inplace=True)


# In[169]:

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_test = random_forest.predict(x_test)
random_forest.score(x_train, y_train)


# In[170]:

result = pd.DataFrame({
        'PassengerId': test_data['PassengerId'],
        'Survived':y_test

    })


# In[171]:

result.to_csv('titanic.csv', index=False)


# In[ ]:



