import sys
from multiprocessing import Process,freeze_support, Manager, Value 
cimport cython
from cython cimport floating
from cpython.exc cimport PyErr_CheckSignals
import interface as bbox
cimport interface as bbox
import neuronet  
cimport neuronet    
cdef float** weights    

cdef neuronet.MLP nn,new_nn
cdef int n_features = 36
cdef int n_actions = 4
cdef int max_time = -1    
cdef float pscore=0 
  
cdef int get_action_by_state(float* state): 
	cdef:
		int best_act = -1
		float best_val = -1e9   
		int act   
	cdef float* output
	
	output=nn.ask(state) 
	if pscore==1:
		new_nn.train(state,output)
	for act in range(4): 
		if(output[act]>best_val):
			best_val = output[act]
			best_act = act 
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
	global nn,new_nn, pscore
	nn=neuronet.MLP(n_features,[16,4],2)
	new_nn=neuronet.MLP(n_features,[36,16,4],3)
	new_nn.set_n(0.1)
	nn.read_weights("weights","source_")
	pscore=1 
	best_res_train=run_bbox("../../levels/train_level.data")    
	pscore=0  
	nn=new_nn
	test=""
	best_res_train=run_bbox("../../levels/train_level.data")
	best_res_test=run_bbox("../../levels/test_level.data") 
	print("RES: Train:"+str(best_res_train)+" Test:"+str(best_res_test)) 
	new_nn.write_weights("weights","dest_") 
	return
	
def run():  
	run_genalg() 
