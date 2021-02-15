import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import preprocessing
import numpy as np 
import matplotlib.pyplot as plt


# Import MoraEbre_RiberaEbre.csv
df = pd.read_csv('/Users/guillemjordibenajes/Desktop/TFG/parsedv2/MoraEbre_RiberaEbre.csv')
df = df.iloc[:1000]
df.insert(19, 'Class', 0) 
print(df.shape)

# Import Gandesa_TerraAlta.csv and append
aux = pd.read_csv('/Users/guillemjordibenajes/Desktop/TFG/parsedv2/Gandesa_TerraAlta.csv')
aux.insert(19, 'Class', 1)  
print(aux.shape)
df = df.append(aux.iloc[:1000])

# Import Tortosa_BaixEbre.csv and append
aux = pd.read_csv('/Users/guillemjordibenajes/Desktop/TFG/parsedv2/Tortosa_BaixEbre.csv') 
aux.insert(19, 'Class', 2) 
print(aux.shape)
df = df.append(aux.iloc[:1000])

# Import Amposta_Montsia.csv and append
aux = pd.read_csv('/Users/guillemjordibenajes/Desktop/TFG/parsedv2/Amposta_Montsia.csv') 
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
X = np.asarray(X, dtype="|S6")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=2)
print(X_train.shape); print(X_test.shape); print(y_train.shape); print(y_test.shape)

min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)
X = min_max_scaler.fit_transform(X)

filename = 'pickle_model.pkl'
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
trial = [0.98593997, 0.99440276, 0.92688337, 0.99778753, 0.47408529, 0.        ]
trial = np.asarray(trial).reshape(1,-1)
#clf2 = pickle.loads(s)
mlp = loaded_model.predict(trial)
print(mlp)