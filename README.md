# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Prepare your data Clean and format your data Split your data into training and testing sets

2.Define your model Use a sigmoid function to map inputs to outputs Initialize weights and bias terms

3.Define your cost function Use binary cross-entropy loss function Penalize the model for incorrect predictions

4.Define your learning rate Determines how quickly weights are updated during gradient descent

5.Train your model Adjust weights and bias terms using gradient descent Iterate until convergence or for a fixed number of iterations

6.Evaluate your model Test performance on testing data Use metrics such as accuracy, precision, recall, and F1 score

7.Tune hyperparameters Experiment with different learning rates and regularization techniques

8.Deploy your model Use trained model to make predictions on new data in a real-world application.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Pooja a
RegisterNumber: 212222240072
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
*/
```

## Output:
### Initial data set:
![238246426-2fb1cd6d-4da0-4245-8087-0f3dc6adbadf](https://github.com/poojaanbu0/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119390329/a91fde4f-a912-4b54-941d-b2ba2e9d7178)

### Data info:
![238246435-b06006a8-34b7-4a52-ad76-bd925c9ce6c3](https://github.com/poojaanbu0/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119390329/3e876e7e-eadc-4125-b21b-6734a505096c)

### Optimization of null values:
![238246551-f585db85-5818-4437-bcc3-2740f900914b](https://github.com/poojaanbu0/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119390329/41bc98ab-231e-4aa5-b285-38e01a826316)

### Assignment of x and y values:
![238246586-f3eddb41-9232-4111-8e3f-648033a2082e](https://github.com/poojaanbu0/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119390329/44e78bdb-9de0-47e2-9434-78881164201a)
![238246606-5afe85e6-d8ca-42bc-8e09-029d80ec2947](https://github.com/poojaanbu0/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119390329/8139955c-c841-445d-b767-b183c6083b30)

### Converting string literals to numerical values using label encoder:
![238246755-46f2a95f-c579-4e6b-a936-5eb265a330fa](https://github.com/poojaanbu0/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119390329/c15456a8-26a2-4964-97b7-b01f177e2d74)

### Accuracy:
![238246796-4a1170b7-70a6-4382-b715-c0b8ef3f5640](https://github.com/poojaanbu0/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119390329/da6ecb46-551a-4091-87ad-d26689596e9a)

### Prediction:
![238246867-404d7a22-42f9-45c2-a40c-34df17264900](https://github.com/poojaanbu0/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/119390329/ccc6ee6a-333d-43ff-a4bc-aa3e594d93cc)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
