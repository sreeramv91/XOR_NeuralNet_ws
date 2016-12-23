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

class NeuralNet:
	def __init__(self,inputs_number,hidden_number,output_number):
		self.inputs_number=inputs_number
		self.hidden_number=hidden_number
		self.output_number=output_number
		
		self.w1 = [rand(-0.2,0.2) for x in range (0,2)]
		self.w2 = [rand(-0.2,0.2) for x in range (0,2)]
		self.w3 = [rand(-0.2,0.2) for x in range (0,2)]
		self.w_out = [rand(-2,2) for x in range (0,2)]
		
	def NeuralNet_Train(self):
		Epoch=1000
		for count in range (0,Epoch):
			for k in range (0,4):
				neto1=0.0
				for i in range (0,self.inputs_number):
					neth[i]=self.w1[i]*input1[k] + self.w2[i]*input2[k] + 1*self.w3[i]
					outh[i]=sigmoid(neth[i])
				for i in range (0,self.hidden_number):
					neto1= neto1 + (self.w_out[i]*outh[i])
				outo1=sigmoid(neto1)
				delta_out=(output[k]-outo1)*dsigmoid(outo1)
				for j in range (0,self.hidden_number):
					delta_Hidden[j]=dsigmoid(outh[j])*delta_out*self.w_out[j]
				for i in range (0,self.inputs_number):
					self.w1[i] = self.w1[i] + learningRate*delta_Hidden[i]*input1[k]
					self.w2[i] = self.w2[i] + learningRate*delta_Hidden[i]*input2[k]
					self.w3[i] = self.w3[i] + learningRate*delta_Hidden[i]*1						#Bias weight being added
					self.w_out[i] = self.w_out[i] + learningRate*outh[i]*delta_out

	def NeuralNet_Test(self):
		
		for k in range (0,4):
			for i in range (0,self.inputs_number):
				neth[i]=self.w1[i]*input1[k] + self.w2[i]*input2[k] + 1*self.w3[i]
				outh[i]=sigmoid(neth[i])
			neto1=0.0
			for i in range (0,self.hidden_number):
				neto1= neto1 + (self.w_out[i]*outh[i])
			outo1=sigmoid(neto1)
			
			print ("The req output is:",output[k],"Obtained is:",outo1)
			
if __name__ == "__main__":

	bias_out=1
	bias_hidden=1
	nn=NeuralNet(2,2,1)
	nn.NeuralNet_Train()
	nn.NeuralNet_Test()
