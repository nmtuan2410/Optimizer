
import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist   
(x_train,y_train),(x_test,y_test)= mnist.load_data()
x_train=np.reshape(x_train,(60000,784))/255.0       
x_test= np.reshape(x_test,(10000,784))/255.0
y_train = np.matrix(np.eye(10)[y_train])
y_test = np.matrix(np.eye(10)[y_test])

def sigmoid(x):
    return 1./(1.+np.exp(-x))

def softmax(x):
    return np.divide(np.matrix(np.exp(x)),np.mat(np.sum(np.exp(x),axis=1)))

def Forwardpass(X,Who,bho,Wh,bh,Wo,bo):  
    zho= X@Who.T + bho                    
    ao=sigmoid(zho)
    zh = ao@Wh.T + bh
    a = sigmoid(zh)
    z=a@Wo.T + bo
    o = softmax(z)
    return o
def AccTest(label,prediction):    
    OutMaxArg=np.argmax(prediction,axis=1)
    LabelMaxArg=np.argmax(label,axis=1)
    Accuracy=np.mean(OutMaxArg==LabelMaxArg)
    return Accuracy

learningRate = 0.5
gama=0.9
Epoch=20          
NumTrainSamples=60000
NumTestSamples=10000
NumInputs=784
NumHiddenUnits=512      
NumClasses=10           
#hidden layer1    
vWho=np.zeros((512,NumInputs)) 
vbho=np.zeros((1,NumHiddenUnits)) 
Who=np.matrix(np.random.uniform(-0.5,0.5,(512,NumInputs))) 
bho= np.random.uniform(0,0.5,(1,512))
dWho= np.zeros((NumHiddenUnits,NumInputs))
dbho= np.zeros((1,NumHiddenUnits))
#hidden layer2
vWh=np.zeros((NumHiddenUnits,512))
vbh=np.zeros((1,NumHiddenUnits))
Wh=np.matrix(np.random.uniform(-0.5,0.5,(NumHiddenUnits,512))) 
bh= np.random.uniform(0,0.5,(1,NumHiddenUnits))
dWh= np.zeros((NumHiddenUnits,512))
dbh= np.zeros((1,NumHiddenUnits))
#Output layer
vWo=np.zeros((NumClasses,NumHiddenUnits))
vbo=np.zeros((1,NumClasses))
Wo=np.random.uniform(-0.5,0.5,(NumClasses,NumHiddenUnits)) 
bo= np.random.uniform(0,0.5,(1,NumClasses))
dWo= np.zeros((NumClasses,NumHiddenUnits))
dbo= np.zeros((1,NumClasses))

from IPython.display import clear_output
loss = []
Acc = []
Batch_size = 200  
Stochastic_samples = np.arange(NumTrainSamples)
for ep in range (Epoch):
  np.random.shuffle(Stochastic_samples)
  for ite in range (0,NumTrainSamples,Batch_size): 
  
    Batch_samples = Stochastic_samples[ite:ite+Batch_size]
    x = x_train[Batch_samples,:]
    y=y_train[Batch_samples,:]  
    zho= x@Who.T + bho          
    ao=sigmoid(zho)              
    zh = ao@Wh.T + bh          
    a = sigmoid(zh)
    z=a@Wo.T + bo
    o = softmax(z)
    loss.append(-np.sum(np.multiply(y,np.log10(o))))
    d = o-y   
    dh = d@Wo   
    dhs = np.multiply(np.multiply(dh,a),(1-a))  
    dho=dhs@Wh   
    dhso= np.multiply(np.multiply(dho,ao),(1-ao)) 
    dWo = np.matmul(np.transpose(d),a) 
    dbo = np.mean(d)   
    dWh = np.matmul(np.transpose(dhs),ao)
    dbh = np.mean(dhs)  
    dWho = np.matmul(np.transpose(dhso),x)    
    dbho = np.mean(dhso)  

    vWo=gama*vWo +(1-gama)*dWo
    vbo=gama*vbo +(1-gama)*dbo
    Wo =Wo - (gama*vWo + learningRate*dWo)/Batch_size           
    bo =bo - (gama*vbo +learningRate*dbo)

    vWh=gama*vWh +(1-gama)*dWh
    vbh=gama*vbh +(1-gama)*dbh
    Wh =Wh-(gama*vWh + learningRate*dWh)/Batch_size
    bh =bh-(gama*vbh + learningRate*dbh)

    vWho=gama*vWho +(1-gama)*dWho
    vbho=gama*vbho +(1-gama)*dbho
    Who =Who-(gama*vWho + learningRate*dWho)/Batch_size
    bho =bho-(gama*vbho + learningRate*dbho)
  
  prediction = Forwardpass(x_test,Who,bho,Wh,bh,Wo,bo)   
  Acc.append(AccTest(y_test,prediction))                 
  print('Epoch:', ep )
  print('Accuracy:',AccTest(y_test,prediction) )

prediction = Forwardpass(x_test,Who,bho,Wh,bh,Wo,bo)
Rate = AccTest(y_test,prediction)
print(Rate)