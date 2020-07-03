import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from joelnet.train import train
from joelnet.nn import NeuralNet
from joelnet.loss import MSE, Log_loss
from joelnet.layers import Linear, Tanh, Sigmoid, reLu
from joelnet.data import BatchIterator
from joelnet.optim import SGD, RMSProp, SGD_Nesterov, Adam, Barzilai, LM, LM_cond
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, auc, precision_score

data = pd.read_csv("../../Datasets/creditcard.csv")

pd.set_option("display.float", "{:.2f}".format)

X = data.drop('Class', axis=1)
y = data.Class

x_sc = StandardScaler()
X_std = x_sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)

y_train = np.array(y_train)
y_test = np.array(y_test)


inputs = X_train
targets = np.array(y_train)
np.random.seed(2)
net = NeuralNet([
    Linear(input_size=30, output_size=24, weights = np.random.randn(30, 24), biases = np.random.randn(24)),
    Tanh(),
    Linear(input_size=24, output_size=30, weights = np.random.randn(24, 30), biases = np.random.randn(30)),
    Tanh(),
    Linear(input_size=30, output_size=35, weights = np.random.randn(30, 35), biases = np.random.randn(35)),
    Tanh(),
    Linear(input_size=35, output_size=1, weights = np.random.randn(35, 1), biases = np.random.randn(1)), 
    Sigmoid()
])

# net = NeuralNet([
#     Linear(input_size=30, output_size=2),
#     Tanh(),
#     Linear(input_size=2, output_size=1), 
#     Sigmoid()
# ])


n_epochs = 20
#loss_list = train(net, inputs,targets, loss = MSE() ,optimizer = Adam(lr = 1e-2, gamma1 = 0.3, gamma2 = 0.3), iterator = BatchIterator(1024), num_epochs = n_epochs)
try:
	loss_list = train(net, inputs,targets, loss = MSE() ,optimizer = Adam(lr = 1e-2, gamma1 = 0.3, gamma2 = 0.25), iterator = BatchIterator(32), num_epochs = n_epochs)
except np.linalg.LinAlgError as err:
	print('Interrompido por matriz singular')

y_pred = []
for x in X_test:
    y_pred.append(net.forward(x))
y_pred = np.array(y_pred)


plt.title("Erro quadrático x Tempo")
plt.xlabel("número de iterações")
plt.ylabel("erro quadrático")
plt.scatter(list(range(0, n_epochs)),loss_list)
plt.savefig(f'Figuras/Fraud/Adam_IMP.png', format='png')
plt.show()


precision, recall, _ = precision_recall_curve(y_test, y_pred)

auc_val = auc(recall, precision)

conf_mat = confusion_matrix(y_test, y_pred.round())
print(f'Confusion matrix: \n( TN | FP)\n(FN | TP) \n\n{conf_mat}')
with open("Evaluation/Adam_imp.txt", "w") as out_file:
	out_str = f"Treinando com {n_epochs} epochs\nConfusion matrix: \n"
	out_str += str(conf_mat)
	out_str += f"\nAUPRC: {auc_val}"
	out_file.write(out_str)


print(f'\n\nAUPRC: {auc_val}')
plt.plot(recall, precision, marker = ".", markersize = 5, label = f"AUC: {auc_val:.3f} ")
plt.title(f'Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.savefig(f'Figuras/Fraud/Adam.png', format='png')
plt.show()

