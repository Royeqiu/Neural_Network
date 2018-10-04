import numpy as np
def softmax(x,der=False):
    if der:
        return np.exp(x)*(np.sum(np.exp(x))-np.exp(x)) / np.power(np.sum(np.exp(x)),2)
    e = np.exp(x - np.max(x, axis=1).reshape((-1, 1)))
    return e / e.sum(axis=1).reshape((-1, 1))
num = np.asarray([[0.3,0.4,0.5]])

def pur_softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def der(x):
    for vector in x:
        lis = []
        for i,value in enumerate(vector):
            value = 1
            for j,comp in enumerate(vector):
                if i==j:
                    value*=pur_softmax(vector)[i]*(1-pur_softmax(vector))[i]
                if i!=j:
                    value*=(-1*pur_softmax(vector)[i]*pur_softmax(vector)[j])

            print(value)
            lis.append(value)
    return lis
print(softmax(num))
print(softmax(num,True))
print(der(num))

print(np.asarray([[1,2,3],[4,5,6]])*np.asarray([[4,5,6],[7,8,9]]))