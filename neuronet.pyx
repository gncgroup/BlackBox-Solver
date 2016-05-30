from libc.stdlib cimport malloc, free
from libc.math cimport log	
from libc.math cimport exp 
from libc.math cimport tanh
from libc.math cimport abs 
import numpy as np 
cimport numpy as np
cimport cython
import copy
import random  
import os.path 


cdef int* int_arr_init(int size):
	cdef int* array
	array = <int*> malloc(sizeof(int)*size)
	for i in range(size):
		array[i] = 0
	return array

cdef float* float_arr_init(int size):
	cdef float* array
	array = <float*> malloc(sizeof(int)*size)
	for i in range(size):
		array[i] = 0
	return array

cdef float** float_arr_init_2d(int size_x,int size_y):
	cdef float** array
	array=<float**> malloc(sizeof(float *)*size_x) 
	for i in range(size_x):
		array[i] = float_arr_init(size_y)
	return array
	
cdef float** float_arr_init_2d_p(int size_x,int* size_y):
	cdef float** array
	array=<float**> malloc(sizeof(float *)*size_x) 
	for i in range(size_x):
		array[i] = float_arr_init(size_y[i])
	return array


cdef float*** float_arr_init_3d(int size_x,int* size_y,int* size_z):
	cdef float*** array
	array=<float***> malloc(sizeof(float **)*size_x);
	for i in range(size_x):
		array[i] = float_arr_init_2d(size_y[i],size_z[i]);
	return array;

cdef float* fprop_compute(float* vector,float*** weights,float** res,int* layers,int layers_count):
	cdef float sum
	res[0]=vector
	
	for l in range(layers_count):
		for n in range(layers[l+1]):
			sum=weights[l][n][layers[l]] #BIAS	
			for w in range(layers[l]):
				sum+=res[l][w]*weights[l][n][w]
			res[l+1][n]=1/(1+exp(-sum))
	return res[layers_count]
			
cdef class Evo_NN:
	def __init__(self,input_length,layers,number_of_layers):
		self.n=1
		self.number_of_layers=number_of_layers
		self.layers=int_arr_init(number_of_layers+1)
		self.layers[0]=input_length
		for i in range(number_of_layers):
			self.layers[i+1]=layers[i]
		self.res=float_arr_init_2d_p(self.number_of_layers+1,self.layers)
		self.init_weights()
	
	def compute(self,vector):
		cdef float* p_vector
		cdef float* out
		p_vector=float_arr_init(self.layers[0])
		for i in range(self.layers[0]):
			p_vector[i]=vector[i]
		out=self.ask(p_vector) 
		return [out[i] for i in range(self.layers[self.number_of_layers])]
	cdef float* ask(self,float* vector): 
		return fprop_compute(vector,self.weights,self.res,self.layers,self.number_of_layers) 

	def init_weights(self):  
		f_weights=[]
		for l in range(self.number_of_layers):
			f_weights.append(np.random.rand(self.layers[l+1],self.layers[l]+1)-1)
				
		self.weights=float_arr_init_3d(self.number_of_layers,self.get_neurons_on_layers(),self.get_weights_size_on_layers())
		for l in range(self.number_of_layers):
			for n in range(self.layers[l+1]): 
				for w in range(self.layers[l]+1):
					self.weights[l][n][w]=copy.deepcopy(f_weights[l][n][w])

	def read_weights(self,path,prefix=""): 
		global weights
		save=False 
		for l in range(self.number_of_layers):
			full_path=path+"/"+prefix+str(l)+".w"
			if os.path.isfile(full_path): 
				f_weights = np.loadtxt(full_path,dtype = np.float32).reshape(self.layers[l+1],self.layers[l]+1) 
				for n in range(self.layers[l+1]):
					for w in range(self.layers[l]+1):    
						self.weights[l][n][w]=copy.deepcopy(f_weights[n,w])  
						
	def write_weights(self,path,prefix=""): 
		global weights 
		for l in range(self.number_of_layers):
			full_path=path+"/"+prefix+str(l)+".w" 
			f=open(full_path,'w')  
			for n in range(self.layers[l+1]):
				for w in range(self.layers[l]+1):  
					f.write(str(self.weights[l][n][w])+"\n")
			f.close()
					
	cdef void set_n(self,float n):
		self.n=n
					
	cdef int* get_neurons_on_layers(self):
		cdef int* neurons_on_layers
		neurons_on_layers=int_arr_init(self.number_of_layers)
		for i in range(self.number_of_layers):
			neurons_on_layers[i]=self.layers[i+1]
		return neurons_on_layers 
		
	cdef int* get_weights_size_on_layers(self):
		cdef int* neurons_on_layers
		neurons_on_layers=int_arr_init(self.number_of_layers)
		for i in range(self.number_of_layers):
			neurons_on_layers[i]=self.layers[i]+1
		return neurons_on_layers 		