import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
    y = 1/(1+np.exp(-x))
    return y

inputx = np.array(([0.11,0.19,0.25],[0.1,0.2,0.15]))
outputy = np.array(([0.89,0.11],[0.9,0.1]))
size_weight = len(inputx[0])
size_output = len(outputy)
size_sample = len(inputx)
thetalayer1 = np.random.randn(size_weight,size_weight)
thetalayer2 = np.random.randn(size_output,size_weight)


b1 = np.array([[0.35 for i in range(size_weight)]])
b2 = np.array([[0.45 for i in range(size_output)]])

go_on = True
i = 0
learningrate=0.2
error = []
iteration = []
while go_on:
    iteration.append(i)
    thetalayer1_full = np.concatenate((thetalayer1,b1.T),axis=1)
    thetalayer2_full = np.concatenate((thetalayer2, b2.T), axis=1)
    error_acc = 0

    stderror = []
    h = []
    a = []
    Zlayer1 = []
    Zlayer2 = []
    for sample_iter in range(size_sample):
        Zlayer1.append(np.matmul(thetalayer1_full, np.concatenate((inputx[sample_iter],[1]))))
        a.append(sigmoid(Zlayer1[sample_iter]))
        Zlayer2.append(np.matmul(thetalayer2_full, np.concatenate((a[sample_iter],[1]))))
        h.append(sigmoid(Zlayer2[sample_iter]))
        stderror.append(h[sample_iter]-outputy[sample_iter])
        error_acc += (1/2*np.sum((stderror[sample_iter])**2))
    error.append(error_acc)
    i += 1
    if error[i-1] <1e-7 or i >20000:
        go_on = False
    else:
        for sample_iter in range(size_sample):
            move2 = np.outer(stderror[sample_iter]*h[sample_iter]*(1-h[sample_iter]),a[sample_iter])
            thetalayer2 = thetalayer2- learningrate * move2/size_sample
            move1 = np.zeros(size_weight)
            for col_move1 in range(size_weight):
                move1[col_move1] = np.sum(stderror[sample_iter]*h[sample_iter]*(1-h[sample_iter])
                                          *thetalayer2[:,col_move1])

            move1 = np.outer(a[sample_iter]*(1-a[sample_iter]),move1*inputx[sample_iter])
            thetalayer1 = thetalayer1- learningrate * move1/size_sample

learningspeed = {'error':error,'iteration':iteration}
lsdf = pd.DataFrame(learningspeed)
lsdf.plot(x='iteration',y='error')

out = h
print(out)


