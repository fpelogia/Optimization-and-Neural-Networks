import numpy as np

def jac_builder (A: np.ndarray , B: np.ndarray) -> np.ndarray:
	res = B.T[0].reshape(B.T[0].shape[0], 1)*A
	for i in range(1,B.shape[1]):
		res = np.hstack((res, B.T[i].reshape(B.T[i].shape[0], 1)*A))
	return res