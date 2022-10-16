import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
data=pd.read_csv("data/data_cleaned.csv")
data.head()

# Regression Tree
def tree_reg(data,columns_x,column_y):
    global y_test_tr
    global X_test_tr
    reg=DecisionTreeRegressor(random_state=1)
    X=data[columns_x]
    Y=data[column_y]
    X_train,X_test_tr,y_train,y_test_tr=train_test_split(X,Y,random_state=1)
    reg.fit(X_train,y_train)
    y_pred=reg.predict(X_test_tr)
    mse=mean_squared_error(y_test_tr,y_pred)
    
    
    return y_pred,mse
data.columns

columns_x=['quantity(kWh)' , 'city',
       'motor_way', 'country_roads', 'consumption(kWh/100km)',
       'A/C', 'park_heating', 'avg_speed(km/h)']
column_y='trip_distance(km)'
y_pred_tr,mse_tr=tree_reg(data,columns_x,column_y)
def plot_graph(y_pred):
    sns.scatterplot(X_test_tr['quantity(kWh)'],y_test_tr)
    sns.scatterplot(X_test_tr['quantity(kWh)'],y_pred)
    plt.show()
def plot_difference(y_pred):
    sns.scatterplot(X_test_tr['quantity(kWh)'],y_test_tr-y_pred)
    plt.show()
plot_graph(y_pred_tr)

plot_difference(y_pred_tr)

data_l=pd.read_csv("data_enc_label.csv")
data_l.columns

columns_xl=['quantity(kWh)' , 'city',
       'motor_way', 'country_roads', 'consumption(kWh/100km)',
       'A/C', 'park_heating', 'avg_speed(km/h)','tire_type_enc', 'driving_style_enc']
column_yl='trip_distance(km)'
y_predl_tr,msel_tr=tree_reg(data_l,columns_xl,column_yl)
plot_graph(y_predl_tr)

plot_difference(y_predl_tr)

data_d=pd.read_csv("data_enc_dummies.csv")
data_d.columns

columns_xd=['quantity(kWh)' , 'city', 'motor_way', 'country_roads', 'consumption(kWh/100km)',
            'A/C', 'park_heating', 'avg_speed(km/h)', 'tire_type_Summer tires', 
            'tire_type_Winter tires',
            'driving_style_Moderate', 'driving_style_Normal']
column_yd='trip_distance(km)'
y_predd_tr,msed_tr=tree_reg(data_d,columns_xd,column_yd)
plot_graph(y_predd_tr)

plot_difference(y_predd_tr)

print("MSE for normal LR is {}".format(mse_tr))
print("MSE for label encoded LR is {}".format(msel_tr))
print("MSE for dummy encoded LR is {}".format(msed_tr))

# Random Forest Regressor
def rf_reg(data,columns_x,column_y):
    global y_test_rf
    global X_test_rf
    reg=RandomForestRegressor(random_state=1)
    X=data[columns_x]
    Y=data[column_y]
    X_train,X_test_rf,y_train,y_test_rf=train_test_split(X,Y,random_state=1)
    reg.fit(X_train,y_train)
    y_pred=reg.predict(X_test_rf)
    mse=mean_squared_error(y_test_rf,y_pred)
    
    
    return y_pred,mse
y_pred_rf,mse_rf=rf_reg(data,columns_x,column_y)
def plot_graph_rf(y_pred):
    sns.scatterplot(X_test_rf['quantity(kWh)'],y_test_rf)
    sns.scatterplot(X_test_rf['quantity(kWh)'],y_pred)
    plt.show()
def plot_difference_rf(y_pred):
    sns.scatterplot(X_test_rf['quantity(kWh)'],y_test_rf-y_pred)
    plt.show()
plot_graph_rf(y_pred_rf)

plot_difference_rf(y_pred_rf)

y_predl_rf,msel_rf=rf_reg(data_l,columns_xl,column_yl)
msel_rf

plot_graph_rf(y_predl_rf)

plot_difference_rf(y_predl_rf)

y_predd_rf,msed_rf=rf_reg(data_d,columns_xd,column_yd)
msed_rf

plot_graph_rf(y_predd_rf)

plot_difference_rf(y_predd_rf)

print("MSE for normal LR is {} km".format(mse_rf))
print("MSE for label encoded LR is {} km".format(msel_rf))
print("MSE for dummy encoded LR is {} km".format(msed_rf))
