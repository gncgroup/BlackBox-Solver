import sys
from multiprocessing import Process,freeze_support, Manager, Value 
cimport cython
from cython cimport floating
from cpython.exc cimport PyErr_CheckSignals
import interface as bbox
cimport interface as bbox
import neuronet  
cimport neuronet   
cdef neuronet.MLP nn
cdef float** weights    

path="/home/username/"		
wpath=path+"train/"			#Path to the script dir
levels_path=path+"levels/"	#Path to the Levels  

neuronet_size=[4] 			#NN Size
eta=20						#L1 penalty size

cdef int n_features = 36
cdef int n_actions = 4
cdef int max_time = -1 

#Sublevels start position
test_set_sublevels=[146631,259142,401800,557441,750809,873755,979972,1035048,1136112]
train_set_sublevels=[149257,308751,469050,625520,777567,839881,946102,1024054,1120743]

cdef int get_action_by_state(float* state): 
	cdef:
		int best_act = -1
		float best_val = -1e9   
		int act   
	cdef float* output	
	output=nn.ask(state) 
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
		
def run_bbox(filename,test_set=False,shared_params=None,shared_list=None):
		global nn
		cdef:
				float* state
				int action, has_next = 1
				int it=0,j=0,inc=0
				float* correct_output
				float* output
				float prev_score=0
		prepare_bbox(filename)
		while has_next:
				state = bbox.c_get_state()
				action = get_action_by_state(state)
				has_next = bbox.c_do_action(action)
				it+=1
				if(test_set==False and it in train_set_sublevels):
						score=bbox.get_score();
						shared_list.append(score-prev_score)
						prev_score=score
		bbox.finish(verbose=0) 
		shared_params.value=bbox.get_score() 
		return bbox.get_score()
		
def test_in_parallel():
		manager = Manager()
		tr = manager.Value('te', 0.0)
		te = manager.Value('tr', 0.0)
		trl = manager.list()
		tel = manager.list()
		shared_params=[0,0]
		if __name__ == 'bot':
				p1 = Process(target=run_bbox,args=(levels_path+"train_level.data",False,tr,trl,))
				p2 = Process(target=run_bbox,args=(levels_path+"test_level.data",True,te,tel,))
				p1.start()
				p2.start()
				p1.join()
				p2.join()
		return [tr.value,te.value,trl,tel]
		
def test_nn(x): 
	i=0
	
	# L1 norm
	weights_sum=0.0
	for l in range(len(neuronet_size)):
		for n in range(neuronet_size[l]):
			for w in range(n_features+1 if l==0 else neuronet_size[l-1]+1):
				nn.weights[l][n][w]=x[i]
				weights_sum+=abs(x[i])
				i+=1 
				
	res_arr=test_in_parallel()
	res,res_test=res_arr[0],res_arr[1]
	
	# Score-function
	score=(min(res_arr[2])-eta*weights_sum)
	print(str(score)+" 0")
	
	nn.write_weights(wpath+"weights/test",str(res)+"_"+str(res_test)) 
	return score
	
def run_genalg():  
	global nn
	nn=neuronet.MLP(n_features,neuronet_size,len(neuronet_size)) 
	arguments=[]
	with open(wpath+"weights/args/"+sys.argv[1]+".a") as f:
		for line in f:
			if line!="":
				arguments.append(float(line))
	test_nn(arguments)
	return
	
def run():
	run_genalg() 

