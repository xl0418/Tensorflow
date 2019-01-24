import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

inputx = np.array([0.5,0.4,0.3])
outputy = np.array([0.01,0.02,0.97])

size_weight = len(inputx)
thetalayer1 = np.random.randn(size_weight,size_weight)
thetalayer2 = np.random.randn(size_weight,size_weight)


b1 = np.array([[0.35 for i in range(size_weight)]])
b2 = np.array([[0.45 for i in range(size_weight)]])

go_on = True
i = 0
learningrate=0.5
error = []
iteration = []
while go_on:
    iteration.append(i)
    thetalayer1_full = np.concatenate((thetalayer1,b1.T),axis=1)
    Zlayer1 =np.matmul(thetalayer1_full, np.concatenate((inputx,[1])))
    a = sigmoid(Zlayer1)
    thetalayer2_full = np.concatenate((thetalayer2,b2.T),axis=1)
    Zlayer2 = np.matmul(thetalayer2_full, np.concatenate((a,[1])))
    h = sigmoid(Zlayer2)
    stderror = h-outputy
    error.append(1/2*np.sum((stderror)**2))
    i += 1
    if error[i-1] <1e-7 or i >20000:
        go_on = False
    else:
        move2 = np.outer(stderror*h*(1-h),a)
        thetalayer2 = thetalayer2- learningrate * move2
        move1 = np.zeros(size_weight)
        for col_move1 in range(size_weight):
            move1[col_move1] = np.sum(stderror*h*(1-h)*thetalayer2[:,col_move1])

        move1 = np.outer(a*(1-a),move1*inputx)
        thetalayer1 = thetalayer1- learningrate * move1

learningspeed = {'error':error,'iteration':iteration}
lsdf = pd.DataFrame(learningspeed)
lsdf.plot(x='iteration',y='error')

out = h
print(out)


