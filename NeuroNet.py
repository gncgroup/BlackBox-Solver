import random
import math
import os.path
import numpy as np
import sys

class Evo_NN: 
	input_length=0
	outputs_length=0 
	
	def __init__(self,input_length,layers,layers_count):
		self.n=1
		self.layers_count=layers_count
		self.layers=[]
		self.layers.append(input_length)
		for i in range(layers_count):
			self.layers.append(layers[i])
		self.init_weights()  
 	 
	def compute(self,vector):    
		sum=0;
		res=[]
		for l in range(self.layers_count+1):
			res.append(np.empty(self.layers[l])) 
			
		res[0]=vector
		for l in range(self.layers_count): 
			res[l]=np.append(res[l],[1])
			for n in range(self.layers[l+1]):
				sum=0
				sum+=np.dot(res[l],self.weights[l][n])   
				res[l+1][n]=1/(1+math.exp(-sum)) 
		return res[self.layers_count]

	def read_weights(self,path,prefix=""): 
		global weights
		save=False 
		for l in range(self.layers_count):
			full_path=path+"/"+prefix+str(l)+".w"
			if os.path.isfile(full_path): 
				self.weights[l] = np.loadtxt(full_path,dtype = np.float32).reshape(self.layers[l+1],self.layers[l]+1)    
					
	def write_weights(self,path,prefix=""): 
		global weights 
		for l in range(self.layers_count):
			full_path=path+"/"+prefix+str(l)+".w" 
			f=open(full_path,'w') 
			for n in range(self.layers[l+1]):
				for w in range(self.layers[l]+1):  
					f.write(str(self.weights[l][n][w])+"\n")
			f.close()
			
	def init_weights(self):  
		self.weights=[]
		for l in range(self.layers_count):
			self.weights.append(np.random.rand(self.layers[l+1],self.layers[l]+1)-0.5) 
					
	def get_neurons_on_layers(self):
		neurons_on_layers=[]
		for i in range(self.layers_count):
			neurons_on_layers.append(self.layers[i+1])
		return neurons_on_layers 
		
	def get_weights_size_on_layers(self):
		neurons_on_layers=[]
		for i in range(self.layers_count):
			neurons_on_layers.append(self.layers[i]+1)
		return neurons_on_layers 			



class Recurrent_Evo_NN:  
	outputs_length=0 
	delta_inputs=36
	recurrent_inputs=4
	transform_inputs=1
	delta_inputs_arr=np.zeros(delta_inputs)
	recurrent_inputs_arr=np.zeros(recurrent_inputs)
	tranform_inputs_arr=np.zeros(transform_inputs) 
	
	def __init__(self,input_length,layers,layers_count):
		self.n=1
		self.layers_count=layers_count
		self.layers=[]
		self.layers.append(input_length+self.delta_inputs+self.recurrent_inputs+self.transform_inputs)  
		for i in range(layers_count):
			self.layers.append(layers[i])
		self.init_weights()  
 	 
	def compute(self,vector):    
		sum=0;
		res=[]
		for l in range(self.layers_count+1):
			res.append(np.zeros(self.layers[l])) 
		
		self.tranform_inputs_arr[0]=abs(vector[35]) 
		res[0]=np.concatenate((np.array(vector),np.array(vector)-self.delta_inputs_arr,self.tranform_inputs_arr,self.recurrent_inputs_arr),axis=0)
		self.delta_inputs_arr=np.array(vector)
		for l in range(self.layers_count): 
			res[l]=np.append(res[l],[1])
			for n in range(self.layers[l+1]):
				sum=0
				sum+=np.dot(res[l],self.weights[l][n])   
				res[l+1][n]=1/(1+math.exp(-sum)) 
		self.recurrent_inputs_arr[0]=res[self.layers_count][0] 
		self.recurrent_inputs_arr[1]=res[self.layers_count][1] 
		self.recurrent_inputs_arr[2]=res[self.layers_count][2] 
		self.recurrent_inputs_arr[3]=res[self.layers_count][3] 
		return res[self.layers_count]

	def read_weights(self,path,prefix=""): 
		global weights
		save=False  
		for l in range(self.layers_count):
			full_path=path+"/"+prefix+str(l)+".w"
			if os.path.isfile(full_path): 
				self.weights[l] = np.loadtxt(full_path,dtype = np.float32).reshape(self.layers[l+1],self.layers[l]+1)    
					
	def write_weights(self,path,prefix=""): 
		global weights 
		for l in range(self.layers_count):
			full_path=path+"/"+prefix+str(l)+".w" 
			f=open(full_path,'w') 
			for n in range(self.layers[l+1]):
				for w in range(self.layers[l]+1):  
					f.write(str(self.weights[l][n][w])+"\n")
			f.close()
			
	def init_weights(self):  
		self.weights=[]
		for l in range(self.layers_count):
			self.weights.append(np.random.rand(self.layers[l+1],self.layers[l]+1)-0.5) 
					
	def get_neurons_on_layers(self):
		neurons_on_layers=[]
		for i in range(self.layers_count):
			neurons_on_layers.append(self.layers[i+1])
		return neurons_on_layers 
		
	def get_weights_size_on_layers(self):
		neurons_on_layers=[]
		for i in range(self.layers_count):
			neurons_on_layers.append(self.layers[i]+1)
		return neurons_on_layers 						