
import numpy as np 
import math
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

lr=0.0000005
beta1=0.9
beta2=0.999
epsilon = 1e-8
t=1
Epoch=50      
NumTrainSamples=60000
NumTestSamples=10000
NumInputs=784
NumHiddenUnits=512      
NumClasses=10           
#hidden layer1    
vhow=np.zeros((NumHiddenUnits,NumInputs))
vhob=np.zeros((1,NumHiddenUnits))
show=np.zeros((NumHiddenUnits,NumInputs))
shob=np.zeros((1,NumHiddenUnits))
Who=np.matrix(np.random.uniform(-0.5,0.5,(512,NumInputs))) 
bho= np.random.uniform(0,0.5,(1,512))
dWho= np.zeros((NumHiddenUnits,NumInputs))
dbho= np.zeros((1,NumHiddenUnits))
#hidden layer2
vhw=np.zeros((NumHiddenUnits,512))
vhb=np.zeros((1,NumHiddenUnits))
shw=np.zeros((NumHiddenUnits,512))
shb=np.zeros((1,NumHiddenUnits))
Wh=np.matrix(np.random.uniform(-0.5,0.5,(NumHiddenUnits,512))) 
bh= np.random.uniform(0,0.5,(1,NumHiddenUnits))
dWh= np.zeros((NumHiddenUnits,512))
dbh= np.zeros((1,NumHiddenUnits))
#Output layer
vw=np.zeros((NumClasses,NumHiddenUnits))
vb=np.zeros((1,NumClasses))
sw=np.zeros((NumClasses,NumHiddenUnits))
sb=np.zeros((1,NumClasses))
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
    # backpropagation 
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
    
    vw=beta1*vw + (1-beta1)*dWo
    vb=beta1*vb + (1-beta1)*dbo
    vcw=vw / (1-beta1**t)
    vcb=vb / (1-beta1**t)
    sw=beta2*sw + (1-beta2)*(np.square(dWo))
    sb=beta2*sb + (1-beta2)*(np.square(dbo))
    scw=sw / (1-beta2**t)
    scb=sb / (1-beta2**t)
    Wo =Wo - (lr*vcw/(np.square(scw)+epsilon))
    bo =bo - (lr*vcb/(np.square(scb)+epsilon))

    vhw=beta1*vhw + (1-beta1)*dWh
    vhb=beta1*vhb + (1-beta1)*dbh
    vhcw=vhw / (1-beta1**t)
    vhcb=vhb / (1-beta1**t)
    shw=beta2*shw + (1-beta2)*(np.square(dWh))
    shb=beta2*shb + (1-beta2)*(np.square(dbh))
    shcw=shw / (1-beta2**t)
    shcb=shb / (1-beta2**t)
    Wh =Wh - (lr*vhcw/(np.square(shcw)+epsilon))
    bh =bh - (lr*vhcb/(np.square(shcb)+epsilon))

    vhow=beta1*vhow + (1-beta1)*dWho
    vhob=beta1*vhob + (1-beta1)*dbho
    vhocw=vhow / (1-beta1**t)
    vhocb=vhob / (1-beta1**t)
    show=beta2*show + (1-beta2)*(np.square(dWho))
    shob=beta2*shob + (1-beta2)*(np.square(dbho))
    shocw=show / (1-beta2**t)
    shocb=shob / (1-beta2**t)
    Who =Who - (lr*vhocw/(np.square(shocw)+epsilon))
    bho =bho - (lr*vhocb/(np.square(shocb)+epsilon))
    
    t=t+1
  prediction = Forwardpass(x_test,Who,bho,Wh,bh,Wo,bo)   
  Acc.append(AccTest(y_test,prediction))                 
  print('Epoch:', ep )
  print('Accuracy:',AccTest(y_test,prediction) )

prediction = Forwardpass(x_test,Who,bho,Wh,bh,Wo,bo)
Rate = AccTest(y_test,prediction)
print(Rate)