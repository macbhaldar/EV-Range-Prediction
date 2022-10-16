import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import seaborn as sns
data=pd.read_csv("data_cleaned.csv")
X=data.drop(['fuel_note','manufacturer', 'model', 'version', 'fuel_date', 'odometer',
       'trip_distance(km)', 'fuel_type','tire_type', 'driving_style',],axis=1)
Y=data['trip_distance(km)']
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=1)
svr=SVR()
svr.fit(X_train,y_train)

y_pred=svr.predict(X_test)
y_pred-y_t

