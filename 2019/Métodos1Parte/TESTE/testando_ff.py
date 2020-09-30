import numpy as np

func = lambda x: max(x,0.0)
func = np.vectorize(func)

def feedforward(I,w1,w2):
    H_unac = np.dot(w1,I) # unac indica que é a versao não ativada da camada
    H = func(H_unac)
    O_unac = np.dot(w2,H)
    O = func(O_unac)
    return O

w1 = np.matrix([[1],[2]])
w2 = np.matrix([3,4])
x = np.matrix([1])

for i in range(6):
	if i != 0:
		print('ff({}) = {}'.format(i,feedforward(i, w1,w2)))





		