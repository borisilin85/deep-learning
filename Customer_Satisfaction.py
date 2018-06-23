import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from os import getcwd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score,r2_score
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor, export_graphviz
import warnings
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor,BaggingRegressor,ExtraTreesRegressor,RandomTreesEmbedding,IsolationForest
from keras.models import Model,Sequential
from keras.layers import Input, Dense,Dropout
import seaborn as sns
from sklearn.feature_selection import SelectKBest,f_classif,chi2
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from keras.callbacks import ModelCheckpoint
import json
from keras.models import model_from_json, load_model

########################################## Pre Processong Feature selection#############################################



train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')
test_data_for_model=test_data.ix[:,1:]
train_data[train_data<0]=0
X_train=train_data.ix[:,1:-1]
Y_train=train_data['TARGET']
col=X_train.columns

scaler = StandardScaler()
X_train=scaler.fit_transform(X_train)
X_train_for_model=pd.DataFrame(X_train,columns=col)
X_test=scaler.fit_transform(test_data_for_model)
X_test_for_model=pd.DataFrame(X_test,columns=col)

number_features=X_train_for_model.shape[1]

#Y_train=np_utils.to_categorical(Y_train)
#columns=train_data.columns[0]
#correlation_matrix= train_data.corr()
#correlation_matrix.to_csv('correlation_matrix.csv')
# mask=SelectKBest(chi2,k=number_features).fit(X_train,Y_train).get_support()
# columns_for_model=X_train.columns[mask]
# #print(X_train[columns_for_model].shape)
# X_train_for_model=X_train[columns_for_model]
# X_test_for_model=test_data[columns_for_model]

########################################## Model ######################################################################
cl_weight = {0: 1,1:20}

model=Sequential()
model.add(Dense(number_features,kernel_initializer='normal', activation='relu',input_dim=number_features))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(number_features, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(number_features, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(number_features, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(1,activation='sigmoid'))

learning_rate=0.1
epochs=150
decay_rate = learning_rate / epochs
sgd = SGD(lr=learning_rate, nesterov=False,momentum=0.7,decay=decay_rate)

model.compile(loss='mse', optimizer=sgd, metrics=['accuracy'])
filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]


history=model.fit(X_train_for_model.values, Y_train, epochs=epochs, batch_size=25,class_weight=cl_weight, callbacks=callbacks_list, verbose=2)
with open('model_architecture.json', 'w') as f:
    f.write(model.to_json())

plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
# plt.plot(history.history['val_acc'])
# plt.plot(history.history['val_loss'])

plt.title('model accuracy + model loss')
plt.ylabel('accuracy,loss')
plt.xlabel('epoch')
plt.legend(['acc','loss'], loc='upper left')
plt.draw()
plt.savefig('loss with acc.png')



# with open('model_architecture.json', 'r') as f:
#     model = model_from_json(f.read())
#
# model.load_weights('weights.best.hdf5')



# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train'], loc='upper left')
# plt.draw()
t =model.predict(X_test_for_model.values)

#t=np.argmax(t,1)

t=t.tolist()
t=[i[0] for i in t ]
ser=pd.Series(t)


test_data['TARGET']=ser


test_data[['ID','TARGET']].to_csv('first_submission.csv',index=False)
#plt.show()