cdef class MLP:
	cdef float n
	cdef int* layers
	cdef int number_of_layers		
	cdef float*** weights 
	cdef float** res	
	cdef float* ask(self,float* vector)
	cdef float* train(self,float* vector,float* correct_output)
	cdef int* get_neurons_on_layers(self)
	cdef int* get_weights_size_on_layers(self)
	cdef void set_n(self,float n)