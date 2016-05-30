#include <stdlib.h> 

float* fprop_compute(float* vector,float*** weights,float** res,int* layers,int layers_count)
{
		float sum;
		res[0]=vector;
		
		int l,n,w;
		for(l=0;l<layers_count;l++){
			for(n=0;n<layers[l+1];n++){
				sum=weights[l][n][layers[l]]; //BIAS	
				for(w=0;w<layers[l];w++){
					sum+=res[l][w]*weights[l][n][w];
				}
				res[l+1][n]=1/(1+expf(-sum));
			}
		}
		return res[layers_count];
}


float* backprop_compute(float* vector,float* correct_output,float learning_rate,float*** weights,float** res,int* layers,int layers_count)
{
		float** delta;	
		float sum;
		delta=float_arr_init_2d_p(layers_count+1,layers);
		
		int l,n,pn,w;
		for(l=layers_count;l>0;l--){ 
			for(n=0;n<layers[l];n++){
				delta[l][n]=res[l][n]*(1-res[l][n]);
				if(l==layers_count){
					delta[l][n]*=correct_output[n]-res[l][n]; 
				}else{
					sum=0;
					for(pn=0;pn<layers[l+1];pn++){
						sum+=weights[l][pn][n]*delta[l+1][pn];
					}
					delta[l][n]*=sum;
				}
				for(w=0;w<layers[l-1];w++){
					weights[l-1][n][w]+=learning_rate*res[l-1][w]*delta[l][n];
				}
				weights[l-1][n][layers[l-1]]+=learning_rate*delta[l][n]; //BIAS
			}
		}
		
		int i;
		for(i=0;i<layers_count+1;i++){
			free(delta[i]);
		}
		free(delta);
		
		return res[layers_count];
}
