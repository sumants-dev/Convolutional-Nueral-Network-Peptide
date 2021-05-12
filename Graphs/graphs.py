import seaborn as sns
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

trainingFiles = glob.glob('*_trainingMetrics.csv')
predictionFiles = glob.glob('*_prediction.csv')
plt.rcParams['lines.linewidth'] = 1.75
df = pd.read_csv('Enr_prediction.csv')
value = sum(df['y_test'])/len(df['y_test'])
baseLine = max(value, 1 - value) 
best_in_benchmark = .73
acc = [.73]
labels = ['Cordax']
for trainingFile in trainingFiles:
  df = pd.read_csv(trainingFile)
  df.iloc[0, :] = [0, 0 , 0, 0]
  plt.figure()
  acc.append(df['AccTest'].values[-1])
  plt.plot(df['Idx'], df['AccTest'], '-', label = 'Test Acc')
  plt.plot(df['Idx'], df['AccTrain'], '--', label = 'Train Acc')
  plt.hlines(baseLine,  color = 'black', xmin= 0, xmax= 105, label= 'Random')
  plt.hlines(.73,  color = 'blue', xmin= 0, xmax= 105, label= 'Cordax')
  plt.title('_'.join(trainingFile.split('_')[0:-1]))
  plt.ylim(.5, 1.05)
  plt.xlim(-.5, 101)
  plt.legend(loc='upper left')
  labels.append('_'.join(trainingFile.split('_')[0:-1]))
  plt.savefig(trainingFile.split('.')[0] + '.png')

for predictionFile in predictionFiles:
  plt.figure()
  df = pd.read_csv(predictionFile)
  cf_matrix = confusion_matrix(df['y_test'].values, df['y_pred_full'].values)
  sns.heatmap(cf_matrix, annot=True, fmt='' , cmap='Blues')
  plt.title('_'.join(predictionFile.split('_')[0:-1]))
  plt.savefig(predictionFile.split('.')[0] + '_cfMatrix.png')
  print(predictionFile)
  print(classification_report(df['y_test'].values, df['y_pred_full'].values))

plt.close('all')

df = pd.DataFrame()
df['Method'] = labels
df['Accuracy'] = acc
df = df.sort_values(by = 'Accuracy')
df = df.round(2)
ax = sns.barplot(x="Method", y="Accuracy", data=df)

