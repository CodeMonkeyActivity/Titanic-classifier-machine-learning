import numpy as np
import pandas as pd
import re as re

train = pd.read_csv('train.csv', header = 0, dtype={'Age': np.float64})
test  = pd.read_csv('test.csv' , header = 0, dtype={'Age': np.float64})
full_data = [train, test]


for dataset in full_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].median())

    
train['CategoricalFare'] = pd.qcut(train['Fare'], 4)


for dataset in full_data:
    age_avg 	   = dataset['Age'].mean()
    age_std 	   = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()   
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)
    
train['CategoricalAge'] = pd.cut(train['Age'], 5)


def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
	if title_search:
		return title_search.group(1)
	return ""


for dataset in full_data:
	dataset['Title'] = dataset['Name'].apply(get_title)
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

def model_training(model, x_train, x_test, y_train, y_test):
	print("***************************************************************")
	print(model.clf)
	print("model training...")
	t0 = time.time();
	model.fit(x_train, y_train)#在训练集训练模型
	train_time = time.time() - t0

	t0 = time.time();
	expected = y_test
	predicted = model.predict(x_test)#在测试集进行测试
	test_time = time.time() - t0

	accuracy = accuracy_score(expected, predicted)
	recall = recall_score(expected, predicted, average="binary")
	precision = precision_score(expected, predicted , average="binary")
	f1 = f1_score(expected, predicted , average="binary")
	cm = confusion_matrix(expected, predicted)
	tpr = float(cm[0][0])/np.sum(cm[0])
	fpr = float(cm[1][1])/np.sum(cm[1])

	print(cm)
	print("tpr:%.3f" %tpr)
	print("fpr:%.3f" %fpr)
	print("accuracy:%.3f" %accuracy)
	print("precision:%.3f" %precision)
	print("recall:%.3f" %recall)
	print("f-score:%.3f" %f1)
	print("train_time:%.3fs" %train_time)
	print("test_time:%.3fs" %test_time)
	print("***************************************************************")

#onehotencode

# Feature Selection
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'FamilySize']
train = train.drop(drop_elements, axis = 1)
train = train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)
test  = test.drop(drop_elements, axis = 1)

train = train.values
test  = test.values

x_train = train[0::, 1::]
y_train = train[0::, 0]
x_test = test[0::, 1::]
y_test = test[0::, 0]


model_gbdt = GradientBoostingClassifier(n_estimators=100)
model_lr = LogisticRegression()
model_adaboost = AdaBoostClassifier()
model_rf = RandomForestClassifier()
model_training(model_gbdt, x_train, x_test, y_train, y_test)
model_training(model_lr, x_train, x_test, y_train, y_test)
model_training(model_adaboost, x_train, x_test, y_train, y_test)
model_training(model_rf, x_train, x_test, y_train, y_test)