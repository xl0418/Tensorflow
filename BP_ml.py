import numpy as np
import matplotlib.pyplot as plt
def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y


thetalayer1 = np.random.randn(2,2)
thetalayer2 = np.random.randn(2,2)
x1 = 0.5
x2 = 0.4
o1 = 0.01
o2 = 0.99

b1 = np.array([[0.35,0.35]])
b2 = np.array([[0.45,0.45]])
go_on = True
i = 0
learningrate=0.5
while go_on:
    thetalayer1_full = np.concatenate((thetalayer1,b1.T),axis=1)
    Zlayer1 =np.matmul(thetalayer1_full, np.array([x1,x2,1]))
    a = sigmoid(Zlayer1)
    thetalayer2_full = np.concatenate((thetalayer2,b2.T),axis=1)
    Zlayer2 = np.matmul(thetalayer2_full, np.concatenate((a,[1])))
    h = sigmoid(Zlayer2)
    stderror = h-np.array([o1,o2])
    error = 1/2*np.sum((stderror)**2)
    i += 1
    if error <1e-7 or i >20000:
        go_on = False
    else:
        move2 = np.outer(stderror*h*(1-h),a)
        thetalayer2 = thetalayer2- learningrate * move2
        move1_1 = np.sum(stderror*h*(1-h)*thetalayer2[:,0])
        move1_2 = np.sum(stderror*h*(1-h)*thetalayer2[:,1])

        move1 = np.outer(a*(1-a),np.array([move1_1*x1,move1_2*x2]))
        thetalayer1 = thetalayer1- learningrate * move1

out = h
print(out)


