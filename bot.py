# coding: utf-8
import interface as bbox 	#Его Величество Черный Ящик
import Ensemble				#Ансамбль Нейросетей

#Число переменных состояния и действий
n_features = n_actions = -1

#Нейроансамбль
ensamble=None

#Подготавливаем Ящик	
def prepare_bbox():
	global n_features, n_actions
	if bbox.is_level_loaded():
		bbox.reset_level()
	else:
		bbox.load_level("../../levels/test_level.data", verbose=1)
		n_features = bbox.get_num_of_features()
		n_actions = bbox.get_num_of_actions() 
		
#Рассчитываем ответы нейросетей, узнаем оптимальное действие, согласно состоянию
def get_action_by_state(state):   
	return ensamble.compute(state) 

#Запускаем Ящик	
def run_bbox():
	global ensamble
	has_next = 1 
	prepare_bbox()  
	ensamble=Ensemble.NN_Ensemble(n_features,4,[[36,64,4],[16,4],[16,4],[36,64,4]],n_actions)  
	ensamble.read_weights("weights")
	
	while has_next: 
		state = bbox.get_state() 
		action = get_action_by_state(state)
		has_next = bbox.do_action(action)   
		if(bbox.get_time()%10000==0): 
			print(str(bbox.get_time())+" "+str(bbox.get_score()))
	bbox.finish(verbose=1)
 
 
if __name__ == "__main__": 
	run_bbox()