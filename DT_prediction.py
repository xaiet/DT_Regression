# Import required libraries
import pandas as pd
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
import sklearn
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPClassifier
from sklearn import tree
# Import necessary modules
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import pickle
# Quizá se tienen que especificar las rutas totales a los archivos .csv
# Import MoraEbre_RiberaEbre.csv
df = pd.read_csv('MoraEbre_RiberaEbre.csv')
df = df.iloc[:1000]
df.insert(19, 'Class', 0) 
print(df.shape)

# Import Gandesa_TerraAlta.csv and append
aux = pd.read_csv('Gandesa_TerraAlta.csv')
aux.insert(19, 'Class', 1)  
print(aux.shape)
df = df.append(aux.iloc[:1000])

# Import Tortosa_BaixEbre.csv and append
aux = pd.read_csv('Tortosa_BaixEbre.csv') 
aux.insert(19, 'Class', 2) 
print(aux.shape)
df = df.append(aux.iloc[:1000])

# Import Amposta_Montsia.csv and append
aux = pd.read_csv('Amposta_Montsia.csv') 
aux.insert(19, 'Class', 3) 
print(aux.shape)
df = df.append(aux.iloc[:1000])

meanValue = df['avrg_speed'].mean()
df.insert(20, 'mean_avg_speed', meanValue) 


#dataset = df.iloc[:,[7, 8, 9, 10, 12, 4, 19, 11]]
dataset = df.iloc[:,[7, 8, 9, 10, 12, 19, 11]]
X = dataset.iloc[:,:6]
y = dataset.iloc[:,6]
print(X.dtypes); print(y.dtypes)
y = np.asarray(y, dtype="|S6")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=2)
print(X_train.shape); print(X_test.shape); print(y_train.shape); print(y_test.shape)

min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)
####0ID,1lane_count,2vehicle_fromTaz,3vehicle_toTaz,4avrg_speed,5ini_time,6end_time,7ini_x_pos,8ini_y_pos,9end_x_pos,10end_y_pos,11tripinfo_duration,12tripinfo_routeLength,13tripinfo_timeLoss,14tripinfo_waitingCount,15tripinfo_waitingTime,16tripinfo_arrivalLane,17tripinfo_departLane,18Repetition,19Class

print('Fitting the model...')

#mlp = MLPClassifier(hidden_layer_sizes=(10,10,10,10,10, 10), activation='relu', solver='adam', max_iter=10000, learning_rate='constant', verbose=True, learning_rate_init=0.001) #learning_rate_initdouble, default=0.001
mlp = DecisionTreeRegressor(max_depth=50, random_state=2)

mlp.fit(X_train,y_train)

print('Classifier trained, predicting...')
predict_test = mlp.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix

print('#######')
predict_test = predict_test.astype(np.float64)
y_test = y_test.astype(np.float64)

print(r2_score(y_test,predict_test))
print(mean_squared_error(y_test,predict_test))

mitjana = y_test.mean()
print("""
    Mitjana: %d
""" % (mitjana))
for i in range(0,15):
	print("X=%s, Real=%s, Predicted=%s" % (X_test[i], y_test[i], predict_test[i]))

#423	121	MoraEbre	TerresEbre	14.601476340694003	11052.0	12636.0	33339.34	63546.65	44432.33	66548.46	1585.0	23149.22	202.91	0	0.0	743729610#0_0	153173569_0	1
print(X_train[0][:])
print(y_train[0][:])

t = X_train[0][:]
t = t.reshape(-1, 6)
time_predicted = mlp.predict(t)
print(time_predicted[0])

fig, ax = plt.subplots()
ax.scatter(y_test, predict_test)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

tags = ['Xi', 'Yi', 'Xf', 'Yf', 'routeLength', 'Class']
 
fig, ax = plt.subplots()
ax.set_ylabel('----------')
ax.set_title('Quantitat')
plt.bar(tags, mlp.feature_importances_)
plt.savefig('importance_DT_AvgSpeed.png')
plt.show()
print(mlp.feature_importances_)

pkl_filename = "pickle_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(mlp, file)

