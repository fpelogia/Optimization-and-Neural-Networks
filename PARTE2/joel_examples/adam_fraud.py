import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from joelnet.train import train
from joelnet.nn import NeuralNet
from joelnet.layers import Linear, Tanh, Sigmoid, reLu
from joelnet.data import BatchIterator
from joelnet.optim import SGD, RMSProp, SGD_Nesterov, Adam
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, precision_score

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


inputs = X_train[0:1000]
targets = np.array(y_train[0:1000])



# net = NeuralNet([
#     Linear(input_size=30, output_size=16),
#     reLu(),
#     Linear(input_size=16, output_size=24),
#     reLu(),
#     Linear(input_size=24, output_size=20),
#     reLu(),
#     Linear(input_size=20, output_size=24),
#     reLu(),
#     Linear(input_size=24, output_size=1),
#     Sigmoid(),
#     Linear(input_size=1, output_size=1)
# ])
net = NeuralNet([
    Linear(input_size=30, output_size=24),
    Tanh(),
    Linear(input_size=24, output_size=30),
    Tanh(),
    Linear(input_size=30, output_size=35),
    Tanh(),
    Linear(input_size=35, output_size=1),
    Sigmoid()
])


n_epochs = 200
loss_list = train(net, inputs,targets, optimizer = Adam(lr = 1e-2, gamma1 = 0.3, gamma2 = 0.4), iterator = BatchIterator(128), num_epochs = n_epochs)

y_pred = []
for x in X_test[0:1000]:
    y_pred.append(net.forward(x))
y_pred = np.array(y_pred)


aux = X_test[0:1000]
indices_1 = np.where(aux == 0)
print('fraudes:', indices_1[0])

plt.title("Erro quadrático x Tempo")
plt.xlabel("número de iterações")
plt.ylabel("erro quadrático")
plt.scatter(list(range(0, n_epochs)),loss_list)
#plt.show()


precision, recall, _ = precision_recall_curve(y_test[0:1000], y_pred)

auc_val = auc(recall, precision)
print(f'Confusion matrix: \n( TN | FP)\n(FN | TP) \n{confusion_matrix(y_test[0:1000], y_pred.round())}')

print(f'\n\nAUPRC: {auc_val}')
# plt.plot(recall, precision, marker = ".", markersize = 5, label = f"AUC: {auc_val:.3f} ")
# plt.title(f'Precision-Recall Curve')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.legend(loc='lower left')
# plt.show()

