"""
We use an optimizer to adjust the parameters
of our network based on the gradients computed
during backpropagation
"""
from joelnet.nn import NeuralNet
from joelnet.jacobian import grad_flatten
import numpy as np
import copy

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.001) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        # print('\n\n\n\n\n\n\n')
        # for param, grad in net.params_and_grads():
        #     print(f'param: {param} \ngrad: {grad}')
        # print('\n\n\n\n\n\n\n')
        for param, grad in net.params_and_grads():
            param -= self.lr *grad

            # try:
            #     param -= self.lr *grad
            # except ValueError: 
            #     param -= self.lr *grad[0]
            

class RMSProp(Optimizer):
    def __init__(self, lr: float = 0.001, gamma: float = 0.9, epsilon: float = 1e-8) -> None:
        self.lr = lr
        self.G = np.zeros(100, dtype = object)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, net: NeuralNet) -> None:

        i = 0
        for param, grad in net.params_and_grads():
            #try:
            G_temp = self.gamma*self.G[i] + (1-self.gamma)*(grad**2)
            param -= (self.lr/(np.sqrt(G_temp + self.epsilon)))*grad
            # except ValueError:
            #     G_temp = self.gamma*self.G[i] + (1-self.gamma)*(grad[0]**2)
            #     param -= (self.lr/(np.sqrt(G_temp + self.epsilon)))*grad[0]
                
            self.G[i] = G_temp
            i = i + 1
        
class SGD_Nesterov(Optimizer):
    def __init__(self, lr: float = 0.001, gamma: float = 0.9) -> None:
        self.lr = lr
        self.gamma = gamma

    def step(self, net: NeuralNet) -> None:

        for param, grad, prev in net.params_and_grads_v2():
            temp = copy.deepcopy(param)
            m = param - prev
            #print('prev:', prev)
            self.intermediate_step(net, m, param)

            #print('param:', param)

            #try:
            param -= self.lr *grad
            # except ValueError: 
            #     param -= self.lr *grad[0]

            prev = temp
            

    def intermediate_step(self, net: NeuralNet, m, param) -> None:
        #for param, grad in net.params_and_grads():
        param = np.add(param , self.gamma * m)

        predicted = net.forward(net.curr_batch.inputs)
        grad = net.loss_f.grad(predicted, net.curr_batch.targets)
        net.backward(grad)


class Adam(Optimizer):
    def __init__(self, lr: float = 0.001, gamma1: float = 0.9, gamma2: float = 0.999,
     epsilon: float = 1e-8) -> None:
        self.lr = lr
        self.G = np.zeros(100, dtype = object)
        self.m = np.zeros(100, dtype = object)
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.epsilon = epsilon

    def step(self, net: NeuralNet) -> None:

        i = 0
        for param, grad in net.params_and_grads():
            #try:
            G_temp = (self.gamma2*self.G[i] + (1.0 - self.gamma2)*grad**2.0)/(1.0 - np.power(self.gamma2, net.n_iter+1.0))
            m_temp = (self.gamma1*self.m[i] + (1.0 -self.gamma1)*grad)/(1.0-np.power(self.gamma1, net.n_iter+1.0) )
            param -= (self.lr*self.m[i])/(np.sqrt(self.G[i]) + self.epsilon)
            # except ValueError:
            #     G_temp = (self.gamma2*self.G[i] + (1.0 - self.gamma2)*grad[0]**2.0)/(1.0 - np.power(self.gamma2, net.n_iter+1.0))
            #     m_temp = (self.gamma1*self.m[i] + (1.0 -self.gamma1)*grad[0])/(1.0-np.power(self.gamma1, net.n_iter+1.0) )
            #     param -= (self.lr*self.m[i])/(np.sqrt(self.G[i]) + self.epsilon)
                
            self.G[i] = G_temp
            self.m[i] = m_temp
            i = i + 1



class Barzilai(Optimizer):
 
    def __init__(self) -> None:
        pass

    def step(self, net: NeuralNet) -> None:
        count = 0
        for param, grad, prev, grad_prev in net.params_and_grads_v4():
            #s_k = x_k - x_{k-1}
            #y_k = gradf(x_k) - gradf(x_{k-1})
            s_k = param - prev
            y_k = grad - grad_prev

            count = count + 1

            if np.linalg.norm(s_k) < 1e-5 or np.linalg.norm(y_k) < 1e-5 :
                break


            #print('sk: ', s_k)   
            #print('yk: ', y_k)
            #print('cima: ', np.inner(s_k.flatten(),y_k.flatten()))
            #print('baixo: ', np.inner(s_k.flatten(), s_k.flatten()))

            eta_k = np.linalg.norm(s_k.flatten())/np.inner(s_k.flatten(), y_k.flatten())

            if eta_k < 0:
                eta_k = 1e-4

            print('tudo: ', eta_k)

            param -= eta_k * grad

            return True
            



#Levenberg_Marquardt
class LM (Optimizer):
    def __init__(self) -> None:
        pass

    def step(self, net: NeuralNet) -> None:

        for param, grad, jac in net.params_and_grads_v3():
            lamb = max( np.linalg.norm(grad), 1e-4)
            JTJ = jac.T@jac
            #print(f'jtj:\n{JTJ} \njtjshp: {JTJ.shape}')
            #print(f'grad:\n{grad} \ngradshp: {grad.shape}')
            sh = grad.shape
            d = np.linalg.solve(JTJ + lamb*np.eye(len(JTJ)), -grad.flatten())
            d = d.reshape(sh)
            #print('HAHHA: ', d.reshape(sh))
            #param -= self.lr *grad
            # try:
            #     d = np.linalg.solve(JTJ + lamb*np.eye(len(JTJ)), -grad)
            # except ValueError:
            #     d = np.linalg.solve(JTJ + lamb*np.eye(len(JTJ)), -grad.T)

            #d = -1*self.lr*grad 
            ang =  np.arccos(np.inner(-grad.flatten(), d.flatten()) / (np.linalg.norm(-grad.flatten()) * np.linalg.norm(d.flatten())) )
            print('ângulo em graus: ', np.degrees(ang) )
            param += d
            # try:
            #     param += d
            # except ValueError: 
            #     param += d[0]


class bmsnl (Optimizer):
    def __init__(self) -> None:
        self.sig_min = 1e-3
        self.sig_max = 1e-2
        self.M = 1e-3
        self.gamma = 1e-3

    def step(self, net: NeuralNet) -> None:

        for param, grad, jac in net.params_and_grads_v3():
            lamb = max( np.linalg.norm(grad), 1e-4)
            JTJ = jac.T@jac
            #print(f'jtj:\n{JTJ} \njtjshp: {JTJ.shape}')
            #print(f'grad:\n{grad} \ngradshp: {grad.shape}')
            sh = grad.shape
            d = np.linalg.solve(JTJ + lamb*np.eye(len(JTJ)), -grad.flatten())
            d = d.reshape(sh)
            #print('HAHHA: ', d.reshape(sh))
            #param -= self.lr *grad
            # try:
            #     d = np.linalg.solve(JTJ + lamb*np.eye(len(JTJ)), -grad)
            # except ValueError:
            #     d = np.linalg.solve(JTJ + lamb*np.eye(len(JTJ)), -grad.T)

            #d = -1*self.lr*grad 
            ang =  np.arccos(np.inner(-grad.flatten(), d.flatten()) / (np.linalg.norm(-grad.flatten()) * np.linalg.norm(d.flatten())) )
            print('ângulo em graus: ', np.degrees(ang) )
            param += d
            # try:
            #     param += d
            # except ValueError: 
            #     param += d[0]

