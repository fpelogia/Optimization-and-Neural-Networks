import numpy as np    

X = [np.transpose(np.matrix([1])),
     np.transpose(np.matrix([2])),
     np.transpose(np.matrix([3])),
     np.transpose(np.matrix([4])),
     np.transpose(np.matrix([5]))
     ]


w1 = np.matrix([[1],[2]])
#w1 = np.matrix(np.random.normal(0.0, pow(2, -0.5),(2,1)))
w2 = np.matrix([3,4])
#w2 = np.matrix(np.random.normal(0.0, pow(1, -0.5),(1,2)))
y = [1,4,9,16,25]