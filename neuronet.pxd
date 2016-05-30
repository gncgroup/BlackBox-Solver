cdef int* int_arr_init(int size)
cdef float* float_arr_init(int size)
cdef float** float_arr_init_2d(int size_x,int size_y)
cdef float** float_arr_init_2d_p(int size_x,int* size_y)
cdef float*** float_arr_init_3d(int size_x,int* size_y,int* size_z)
cdef float* fprop_compute(float* vector,float*** weights,float** res,int* layers,int layers_count)
cdef class Evo_NN:
	cdef float n
	cdef int* layers
	cdef int number_of_layers		
	cdef float*** weights 
	cdef float** res	
	cdef float* ask(self,float* vector) 
	cdef int* get_neurons_on_layers(self)
	cdef int* get_weights_size_on_layers(self)
	cdef void set_n(self,float n)