#include <stdlib.h>

int* int_arr_init(int size)
{
    int* array;
    array = malloc(sizeof(int)*size);
    int i;
    for (i=0; i<size; i++)
    {
		array[i] = 0;
    }
    return array;
}

float* float_arr_init(int size)
{
    float* array;
    array = malloc(sizeof(int)*size);
    int i;
    for (i=0; i<size; i++)
    {
		array[i] = 0;
    }
    return array;
}

float** float_arr_init_2d(int size_x,int size_y)
{
    float** array=(float**) malloc(sizeof(float *)*size_x);
    int i;
    for (i=0; i<size_x; i++)
    {
		array[i] = float_arr_init(size_y);
    }
    return array;
} 

float** float_arr_init_2d_p(int size_x,int* size_y)
{
    float** array=(float**) malloc(sizeof(float *)*size_x);
    int i;
    for (i=0; i<size_x; i++)
    {
		array[i] = float_arr_init(size_y[i]);
    }
    return array;
} 



float*** float_arr_init_3d(int size_x,int* size_y,int* size_z)
{
    float*** array=(float***) malloc(sizeof(float **)*size_x);
    int i;
    for (i=0; i<size_x; i++)
    {
		array[i] = float_arr_init_2d(size_y[i],size_z[i]);
    }
    return array;
} 



float* float_arr_concat(int size1,int size2,float* arr1,float* arr2)
{
    float* array=float_arr_init(size1+size2);
	int i;
	for(i=0;i<size1;i++){
		array[i]=arr1[i];
	}
	for(i=0;i<size2;i++){
		array[i+size1]=arr2[i];
	}
    return array;
} 

void float_arr_shift(float* array,int size)
{ 
	int i; 
	for(i=size-1;i>0;i--){
		array[i]=array[i-1];
	}
}