import math
import random

input1 = [0,1,0,1]
input2 = [0,0,1,1]
output = [0,1,1,0]
neth = [0 for x in range (1,4)]
outh = [0 for x in range (1,4)]
neto1=0
outo1=0
delta_out=0
delta_weight_sum = 0
delta_Hidden = [0 for x in range (1,4)]
learningRate=0.5

def sigmoid(x):
		return math.tanh(x)
		
def rand(a, b):
		return (b-a)*random.random() + a
		
def dsigmoid(y):
		return 1.0 - y**2

def NeuralNet_Train(w1,w2,w3,w_out):
	
	for k in range (0,4):
		neto1=0.0
		for i in range (0,2):
			neth[i]=w1[i]*input1[k] + w2[i]*input2[k] + 1*w3[i]
			outh[i]=sigmoid(neth[i])
		for i in range (0,2):
			neto1= neto1 + (w_out[i]*outh[i])
		outo1=sigmoid(neto1)
		delta_out=(output[k]-outo1)*dsigmoid(outo1)
		for j in range (0,2):
			delta_Hidden[j]=dsigmoid(outh[j])*delta_out*w_out[j]
		for i in range (0,2):
			w1[i] = w1[i] + learningRate*delta_Hidden[i]*input1[k]
			w2[i] = w2[i] + learningRate*delta_Hidden[i]*input2[k]
			w3[i] = w3[i] + learningRate*delta_Hidden[i]*1						#Bias weight being added
			w_out[i] = w_out[i] + learningRate*outh[i]*delta_out
			
def NeuralNet_Test(w1,w2,w3,w_out):
	
	for k in range (0,4):
		for i in range (0,2):
			neth[i]=w1[i]*input1[k] + w2[i]*input2[k] + 1*w3[i]
			outh[i]=sigmoid(neth[i])
		neto1=0.0
		for i in range (0,2):
			neto1= neto1 + (w_out[i]*outh[i])
		outo1=sigmoid(neto1)
		
		print ("The req output is:",output[k],"Obtained is:",outo1)
			
if __name__ == "__main__":
	
	w1 = [rand(-0.2,0.2) for x in range (0,2)]
	w2 = [rand(-0.2,0.2) for x in range (0,2)]
	w3 = [rand(-0.2,0.2) for x in range (0,2)]
	w_out = [rand(-2,2) for x in range (0,2)]
	

	bias_out=1
	bias_hidden=1
	Epoch=1000
	for x in range (0,Epoch):
		NeuralNet_Train(w1,w2,w3,w_out)
	NeuralNet_Test(w1,w2,w3,w_out)
