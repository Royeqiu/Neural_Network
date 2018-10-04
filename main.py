import numpy as np
import pprint
from sklearn import datasets
from scipy import special
pp = pprint.PrettyPrinter(indent=4)
feature_size = 4

def softmax(x,der=False):
    if der:
        return np.exp(x)*(np.sum(np.exp(x))-np.exp(x)) / np.power(np.sum(np.exp(x)),2)
    e = np.exp(x - np.max(x, axis=1).reshape((-1, 1)))
    return e / e.sum(axis=1).reshape((-1, 1))

def sigmoid(x,der=False):
    if der:
        return x*(1-x)
    else:
        return special.expit(x)


def generate_batch(data,batch_size):
    data_set = []
    batch = []
    for i in range(0,len(data)):

        if i!=0 and i%batch_size==0:
            data_set.append(np.asarray(batch))
            batch = []
        batch.append(data[i])
    return data_set

def random_sample(input, output):

    validation_input = []
    validation_output = []
    for i in range(0,10):
        picked_number=np.random.randint(0,len(input))
        validation_input.append(input[picked_number])
        validation_output.append(output[picked_number])
        input = np.delete(input, picked_number, 0)
        output = np.delete(output, picked_number, 0)

    return validation_input,validation_output

iris = datasets.load_iris()
input =iris['data'][:,:feature_size]
output = iris['target']



def int_to_vector(data,label_size):
    vector = np.zeros((len(data),label_size),dtype='f')
    for i,single in enumerate(data):
        vector[i][single]=1
    return vector
label_size = 3
second_layer_size = 4
np.random.seed(1)

w1 = 2 * np.random.random((feature_size,second_layer_size))-1 #4,4
w2 = 2 * np.random.random((second_layer_size,label_size))-1 #4,1
epoch = 30000

new_output=[]
for value in output:
    new_output.append([value])
output = np.asarray(new_output)

validation_input,validation_output=random_sample(input,output)
for ep in range(0,epoch):
    label_vector = int_to_vector(output,label_size)
    l1 = input
    l2 = sigmoid(np.dot(l1,w1))
    l3 = softmax(np.dot(l2,w2))
    l3_error = label_vector-l3
    l3_delta = l3_error*softmax(l3,True)
    l2_error = l3_delta.dot(w2.T)
    l2_delta = l2_error*sigmoid(l2,True)
    w2 += l2.T.dot(l3_delta)
    w1 += l1.T.dot(l2_delta)
    if ep%500==0:
        print(np.mean(np.abs(l3_error)))

pp.pprint([np.argmax(x) for x in l3])
pp.pprint(label_vector)

def predict(validation_input,validation_output):
    l1 = validation_input
    l2 = sigmoid(np.dot(l1, w1))
    l3 = softmax(np.dot(l2, w2))
    print([np.argmax(x) for x in l3])
    print([x[0] for x in np.asarray(validation_output)])

predict(validation_input,validation_output)
