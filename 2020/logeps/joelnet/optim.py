from joelnet.nn import NeuralNet
from joelnet.jacobian import grad_flatten
import numpy as np
import copy
import time 

class Optimizer:
    def step(self, net: NeuralNet) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, lr: float = 0.001) -> None:
        self.lr = lr

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():
            param -= self.lr *grad

            

class RMSProp(Optimizer):
    def __init__(self, lr: float = 0.001, gamma: float = 0.9, epsilon: float = 1e-8) -> None:
        self.lr = lr
        self.G = np.zeros(100, dtype = object)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, net: NeuralNet) -> None:

        i = 0
        for param, grad in net.params_and_grads():

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
            lamb = max( np.linalg.norm(grad), 1e-6)
            JTJ = jac.T@jac
            sh = grad.shape
            d = np.linalg.solve(JTJ + lamb*np.eye(len(JTJ)), -grad.flatten())
            d = d.reshape(sh)
            param += d




class GD_cond(Optimizer):
    def __init__(self, lr: float = 0.00001) -> None:
        self.lr = lr
        self.alpha = 1e-2

    def step(self, net: NeuralNet) -> None:
        for param, grad in net.params_and_grads():

            predicted = net.forward(net.curr_batch.inputs)
            loss_old = net.loss_f.loss(predicted, net.curr_batch.targets)
            old_param = copy.deepcopy(param)
            param -= self.lr *grad
            count = 0


            predicted = net.forward(net.curr_batch.inputs)
            loss = net.loss_f.loss(predicted, net.curr_batch.targets)

            

            
            temp_lr = self.lr
            while not loss <= loss_old - self.alpha*(np.linalg.norm(param - old_param))**2:

            	print(f'lr: {temp_lr}')
                
            	temp_lr = temp_lr / 2.0

            	param = old_param - temp_lr *grad

            	predicted = net.forward(net.curr_batch.inputs)
            	loss = net.loss_f.loss(predicted, net.curr_batch.targets)
            	#print(f'\nloss: {loss}\nloss_desejada: {loss_old - self.alpha*(np.linalg.norm(param - old_param))**2}')
            	if temp_lr < 1e-10:
            		print('Passo muito pequeno')
            		break
            	count = count + 1


 

class LM_cond (Optimizer):
    def __init__(self, alpha: float = 1e2, lamb: float = 1e2) -> None:
        self.alpha = alpha
        self.lamb = lamb

    def step(self, net: NeuralNet) -> None:
        count = 0
        for param, grad, jac in net.params_and_grads_v3():
            if count == 0:
                print('Dando o passo para w1 e w2')
            else:
                print('Dando o passo para w3 e w4')
            count = count + 1
            predicted = net.forward(net.curr_batch.inputs)
            loss_old = net.loss_f.loss(predicted, net.curr_batch.targets)


            #lamb = min(max( np.linalg.norm(gf), 1e-5), 1e5)
            #print(f'GRADIENTE: {gf}')
            #print(f'NORMA GRAD: {np.linalg.norm(gf)}')
            lamb = self.lamb
            print('grad: ', grad)
            print('jac: ', jac)
            JTJ = jac.T@jac
            print('jtj: ', JTJ)
            sh = grad.shape
            d = np.linalg.solve(JTJ + lamb*np.eye(len(JTJ)), -grad.flatten())
            d = d.reshape(sh)
            # inner_p = np.inner(-grad.flatten(), d.flatten())
            # print('produto interno: ', inner_p )

            old_param = copy.deepcopy(param)

            param += d

            predicted = net.forward(net.curr_batch.inputs)
            loss = net.loss_f.loss(predicted, net.curr_batch.targets)

            print('param :', param)
            #time.sleep(30)
            lixo = input('oi')
            loop_count = 0
            print(f'erro: {loss} / {loss_old - self.alpha*(np.linalg.norm(param - old_param))**2}')
            while not loss <= loss_old - self.alpha*(np.linalg.norm(param - old_param))**2:
                #print(f'f: {loss_old}')
                #print(f'x: {param}\nx^k: {old_param}')
                #print(f'x-xk: {param - old_param}')
                #print('||x-xk||^2: ',np.linalg.norm(param - old_param)**2)
                
                if loop_count == 0:
                    print('entrou no loop da busca linear')
                lamb = 2*lamb
                
                #print(f'erro: {loss} / {loss_old - self.alpha*(np.linalg.norm(param - old_param))**2}')
                d = np.linalg.solve(JTJ + lamb*np.eye(len(JTJ)), -grad.flatten())
                d = d.reshape(sh)
                param += d
                predicted = net.forward(net.curr_batch.inputs)
                loss = net.loss_f.loss(predicted, net.curr_batch.targets)

                # inner_p = np.inner(-grad.flatten(), d.flatten())
                # print('produto interno: ', inner_p )

                loop_count = loop_count + 1

                if lamb > 1e10 :
                    print('LAMBDA GRANDE')
                    break

            if loop_count > 0:
                print(f'saiu do loop com {loop_count} giros')
            else:
                print('não entrou no loop')
            net.n_eval = net.n_eval + loop_count + 1
