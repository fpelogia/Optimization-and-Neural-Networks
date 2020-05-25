import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, precision_score

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore',category=FutureWarning)

from tensorflow import keras

data = pd.read_csv("creditcard.csv")

pd.set_option("display.float", "{:.2f}".format)

X = data.drop('Class', axis=1)
y = data.Class

x_sc = StandardScaler()
X_std = x_sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.2, random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_test = np.array(y_test)

model = keras.models.Sequential([
    keras.layers.Dense(units=16, input_dim=30, activation="relu"),
    keras.layers.Dense(units=24, activation="relu"),  
    keras.layers.Dropout(0.5),  
    keras.layers.Dense(20, activation="relu"),  
    keras.layers.Dense(24, activation="relu"),  
    keras.layers.Dense(1, activation="sigmoid"),  
])


#opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=False)
model.compile(optimizer = "adam", loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_split = 0.1, batch_size=50, epochs=10)


#PARA SALVAR OS MODELOS E DEPOIS PLOTAR A PRC
#model.save("Adam_bce.h5")

y_pred = model.predict(X_test)
y_test = pd.DataFrame(y_test)

print(f'\naccuracy: {accuracy_score(y_test, y_pred.round())}')

# prec_sc = precision_score(y_test, y_pred.round())

print('lyp:', len(y_pred))
precision, recall, thr = precision_recall_curve(y_test, y_pred)
print('PRC P:', len(precision))
auc_val = auc(recall, precision)
print('thresholds\n', thr)

plt.plot(recall, precision, marker = ".", markersize = 5, label = "HUSDHUAS" + f"     AUC: {auc_val:.3f} " )
plt.title(f'Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')

plt.show()
