cdef extern from "arrayinit.c":
	float* float_arr_init(int size)
	float** float_arr_init_2d(int size_x,int size_y)
	float* float_arr_concat(int size1,int size2,float* arr1,float* arr2)
	void float_arr_shift(float* array,int size)	
from libc.math cimport log	
from libc.math cimport exp
from libc.math cimport tanh
from libc.math cimport fabs
import interface as bbox
cimport interface as bbox
import neuronet  
cimport neuronet  
import numpy as np 
cimport numpy as np
cimport cython
import copy
import random
import sys
import time
from cython cimport floating
import os.path
from cpython.exc cimport PyErr_CheckSignals
from libc.stdlib cimport free
cdef neuronet.MLP nn
cdef int n_features = 36
cdef int n_actions = 4
cdef int max_time = -1
cdef float* prev_state  
cdef float** weights    
cdef int pscore=0
level="" 

cdef int get_action_by_state(float* state): 
	global nn,n_features,pscore
	cdef:
		int best_act = -1
		float best_val = -1e9
		int act
	cdef float* output   
	output=nn.ask(state)      
	for act in [0,1,2,3]: 
		if(output[act]>best_val):
			best_val = output[act] 
			best_act = act  	 
	if fabs(output[2]-output[1])<0.002488 and (best_act in [1,2]): 
		best_act=pscore      				
	pscore=best_act
	return best_act 
def prepare_bbox(filename):
	global n_features, n_actions, max_time
	global level
	if bbox.is_level_loaded() and level==filename:
		bbox.reset_level()
	else:
		level=filename	
		bbox.load_level(filename, verbose=0)
		n_features = bbox.get_num_of_features()
		n_actions = bbox.get_num_of_actions()
		max_time = bbox.get_max_time()
		
def run_bbox(filename):
	cdef:
		float* state
		int action, has_next = 1 
		float* output		
	prepare_bbox(filename)    
	while has_next:
		state = bbox.c_get_state()  
		action = get_action_by_state(state)   
		has_next = bbox.c_do_action(action)    
	bbox.finish(verbose=0) 
	return bbox.get_score()
cdef void run_genalg():  
	global nn,new_nn,n_features 
	nn=neuronet.MLP(n_features,[16,4],2)
	nn.read_weights("weights","") 
	best_res_train=run_bbox("levels/train_level.data") 
	best_res_test=run_bbox("levels/test_level.data") 
	print("RES: Train:"+str(best_res_train)+" Test:"+str(best_res_test)) 
	return
	
def run(): 
	run_genalg() 