import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)

from tensorflow import keras
from sklearn.metrics import precision_recall_curve, auc, precision_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("../../Datasets/creditcard.csv")

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

model_list = []
model_list.append(keras.models.load_model("SGD.h5"))
model_list.append(keras.models.load_model("SGD_Nesterov_bce.h5"))
model_list.append(keras.models.load_model("RMSProp_bce.h5"))
model_list.append(keras.models.load_model("Adam_bce.h5"))

labels = ["SGD","SGD Nesterov","RMSProp","Adam"]
i = 0

for model in model_list:
	y_pred = model.predict(X_test)
	y_test = pd.DataFrame(y_test)

	print(f'\n{labels[i]} accuracy: {accuracy_score(y_test, y_pred.round())}')

	prec_sc = precision_score(y_test, y_pred.round(), average='binary')


	precision, recall, _ = precision_recall_curve(y_test, y_pred)

	auc_val = auc(recall, precision)

	plt.plot(recall, precision, marker = ".", markersize = 5, label = labels[i] + f"     AUC: {auc_val:.3f} " )
	plt.title(f'Precision-Recall Curve')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.legend(loc='lower left')

	i = i + 1
plt.show()

