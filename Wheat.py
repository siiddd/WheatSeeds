#Import Packages
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns
import matplotlib.pyplot as plt

#Import the Wheat Dataset
df = pd.read_csv('https://raw.githubusercontent.com/siiddd/WheatSeeds/master/Wheat.csv')

features = ['Area', 'Perimeter', 'Compactness', 'Length of Kernel','Width of Kernel', 'Asymmetric Coeff.', 'Length of Kernel Groove']

df_features = df[features]
df_class = df['Class']-1


#Build a Neural Network
model=Sequential() 

model.add(Dense(500, input_dim = 7,activation='relu')) #Input layer with 7 Inputs And 500 Neurons in the First Hidden Layer
model.add(Dense(3,activation='sigmoid')) #Output Layer with 3 Outputs


#Compile our model after declaring 'Loss' 'Optimizer'
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
 

#Fit our Data into the model
model.fit(df_features, df_class, epochs = 100, validation_split = 0.10)


#Get Summary about the number of Parameters
model.summary()


#Extract the Metrics from our model to use it to see our model validity and performance
x = model.history
model.history.history.keys() #Loss, Accuracy for Train and Test Data
x.history['accuracy'] #Extract just the Accuracy


#Plot Accuracy vs. Epochs
epochs = x.epoch
accuracy = x.history['accuracy']
sns.lineplot(x = epochs, y = accuracy, markers = True)
graph = plt.gcf()
graph.set_size_inches(15, 15)
graph.savefig(r'C:\Users\nsid4\Desktop\AccuracyVsEpoch.png', DPI = 100)


#Final Model Loss and Accuracy
model.evaluate(df_features, df_class)


