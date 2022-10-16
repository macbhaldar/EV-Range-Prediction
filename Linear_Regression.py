import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("data/data_cleaned.csv")
data.columns

X=data.drop(['fuel_note','manufacturer', 'model', 'version', 'fuel_date', 'odometer',
       'trip_distance(km)', 'fuel_type','tire_type', 'driving_style',],axis=1)
Y=data['trip_distance(km)']
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=1)
lr=LinearRegression()
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)
sns.scatterplot(X_test['quantity(kWh)'],y_test)
sns.scatterplot(X_test['quantity(kWh)'],y_pred)

mse(y_test,y_pred)
lr.score(X_test,y_test)
data_enc_l=pd.read_csv("data_enc_label.csv")
data_enc_l.head()

X_enc_l=data_enc_l.drop(['fuel_note','manufacturer', 'model', 'version', 'fuel_date', 'odometer',
       'trip_distance(km)', 'fuel_type','tire_type', 'driving_style',],axis=1)
Y_enc_l=data_enc_l['trip_distance(km)']
X_train_enc_l,X_test_enc_l,y_train_enc_l,y_test_enc_l=train_test_split(X_enc_l,Y_enc_l,random_state=1)
lr.fit(X_train_enc_l,y_train_enc_l)
y_pred_enc_l=lr.predict(X_test_enc_l)
y_pred_enc_l

sns.scatterplot(X_test_enc_l['quantity(kWh)'],y_test_enc_l)
sns.scatterplot(X_test_enc_l['quantity(kWh)'],y_pred_enc_l)

mse(y_test_enc_l,y_pred_enc_l)
lr.score(X_test_enc_l,y_test_enc_l)
data_enc_dum=pd.read_csv("data_enc_dummies.csv")
data_enc_dum.head()

X_enc_dum=data_enc_dum.drop(['fuel_note','manufacturer', 'model', 'version', 'fuel_date', 'odometer',
       'trip_distance(km)', 'fuel_type'],axis=1)
Y_enc_dum=data_enc_dum['trip_distance(km)']
X_train_enc_dum,X_test_enc_dum,y_train_enc_dum,y_test_enc_dum=train_test_split(X_enc_dum,Y_enc_dum,random_state=1)
lr.fit(X_train_enc_dum,y_train_enc_dum)
y_pred_enc_dum=lr.predict(X_test_enc_dum)
y_pred_enc_dum

sns.scatterplot(X_test_enc_dum['quantity(kWh)'],y_test_enc_dum)
sns.scatterplot(X_test_enc_dum['quantity(kWh)'],y_pred_enc_dum)
plt.legend(["Predicted","Actual"])

mse(y_test_enc_dum,y_pred_enc_dum)
lr.score(X_test_enc_dum,y_test_enc_dum)
mse_n=mse(y_test,y_pred)
mse_enc_l=mse(y_test_enc_l,y_pred_enc_l)
mse_enc_dum=mse(y_test_enc_dum,y_pred_enc_dum)
plt.figure(figsize=(20, 10))
plt.suptitle("Lr model")
plt.subplot(1, 3, 1)
plt.title("Normal")
sns.scatterplot(X_test['quantity(kWh)'],y_test)
sns.scatterplot(X_test['quantity(kWh)'],y_pred)


plt.subplot(1, 3, 2)
plt.title("label encoded")
sns.scatterplot(X_test_enc_l['quantity(kWh)'],y_test_enc_l)
sns.scatterplot(X_test_enc_l['quantity(kWh)'],y_pred_enc_l)


plt.subplot(1, 3, 3)
plt.title("dummies encoded")
sns.scatterplot(X_test_enc_dum['quantity(kWh)'],y_test_enc_dum)
sns.scatterplot(X_test_enc_dum['quantity(kWh)'],y_pred_enc_dum)


plt.show()

plt.figure(figsize=(20, 10))
plt.suptitle("Lr model")
plt.subplot(1, 3, 1)
sns.scatterplot(X_test['quantity(kWh)'],y_test-y_pred)


plt.subplot(1, 3, 2)
sns.scatterplot(X_test_enc_l['quantity(kWh)'],y_test_enc_l-y_pred_enc_l)


plt.subplot(1, 3, 3)
sns.scatterplot(X_test_enc_dum['quantity(kWh)'],y_test_enc_dum-y_pred_enc_dum)



plt.show()

print("MSE for normal LR is {}".format(mse_n))
print("MSE for label encoded LR is {}".format(mse_enc_l))
print("MSE for dummy encoded LR is {}".format(mse_enc_dum))

sns.scatterplot(X_test['quantity(kWh)'],y_test)
sns.scatterplot(X_test['quantity(kWh)'],y_pred)

sns.scatterplot(X_test_enc_l['quantity(kWh)'],y_test_enc_l)
sns.scatterplot(X_test_enc_l['quantity(kWh)'],y_pred_enc_l)

sns.scatterplot(X_test_enc_dum['quantity(kWh)'],y_test_enc_dum)
sns.scatterplot(X_test_enc_dum['quantity(kWh)'],y_pred_enc_dum)
