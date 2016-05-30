# import pyximport; pyximport.install()
from subprocess import call
call(["python3 setup.py build_ext --inplace"], shell=True)
import neuronet as nn
import numpy as np
#NeuroNet Ensemble
class NN_Ensemble:
	NN_num=1		#Neural Networks in Ensemble
	NN_sizes=[]		#Number of NN
	NNs=[]			#NeuroNets
	inp_size=36		#Input dim. size
	output_size=4	#Outputs
	pout=0 
	#Ensemble Initialization
	def __init__(self,inp_size,NN_num,NN_sizes,output_size):
		self.NN_num=NN_num
		self.NN_sizes=NN_sizes
		self.inp_size=inp_size
		self.output_size=output_size
		for i in range(self.NN_num):
			self.NNs.append(nn.Evo_NN(inp_size,NN_sizes[i],len(NN_sizes[i]))) 

	#Reads weights
	def read_weights(self,path):
		for i in range(self.NN_num):
			self.NNs[i].read_weights(path,str(i)+"_")
 
	#Estimate accuracy calculation
	def calc_accuracy(self,output):
		accuracy=sorted(np.sort(output))
		accuracy=accuracy[self.output_size-1]-accuracy[self.output_size-2]
		return accuracy
		
	#Forward Prop
	def compute(self,state):
		output=np.zeros((self.NN_num,4))
		for i in range(self.NN_num):
			output[i]=self.NNs[i].compute(state) 
		main_accuracy=self.calc_accuracy(output[0])	
		accuracy=self.calc_accuracy(output[0])	 
		
		out=output[0].argmax()
		if(accuracy>0.4):   
			return out   
			  
		out=output[3].argmax()
		output[3][3]=output[3][0]=0 
		accuracy=self.calc_accuracy(output[3])	
		if(accuracy>0.95 and out in [1,2]):  
			return out    
			  
		accuracy=self.calc_accuracy(output[1])		
		out=output[1].argmax()
		if((accuracy>0.8 and out!=3 )):  
			return out 
			  
		out=output[2].argmax()
		return out 