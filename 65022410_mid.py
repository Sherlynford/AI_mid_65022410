from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


File_path = 'C:/Users/User/Downloads/'
File_name = 'car_data.csv'

df = pd.read_csv(File_path + File_name)

df.drop(columns=['User ID'], inplace=True)
x = df.iloc[:, 1:3].bfill()
y = df['Purchased']

x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=80)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(x_train, y_train)

feature_names = x.columns.tolist()
class_names = y.unique().tolist()

plt.figure(figsize=(25, 20))
_ = plot_tree(model,
              feature_names=feature_names,
              class_names=class_names,
              label='all',
              impurity=True,
              precision=3,
              filled=True,
              rounded=True,
              fontsize=16)
plt.show()

accuracy_train = model.score(x_train,y_train)
accuracy_test = model.score(x_test, y_test)
print('Accuracy_Train: {:.2f}'.format(accuracy_train))
print('Accuracy_Test: {:.2f}'.format(accuracy_test))

feature_imp = model.feature_importances_
feature_names = ['Age','AnnualSalary']

sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(x=feature_imp, y=feature_names)