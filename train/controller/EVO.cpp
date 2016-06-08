#include <shark/Algorithms/DirectSearch/CMA.h>
#include <shark/ObjectiveFunctions/Benchmarks/Benchmarks.h>

using namespace shark; 

std::string path="/home/username/train/";	//Path to the agent directory
int nn_layers=1;							//Number of NN layers

// Executes a shell command
std::string exec(const char* cmd) {
    char buffer[128];
    std::string result = "";
    FILE* pipe = popen(cmd, "r");
    if (!pipe) throw std::runtime_error("popen() failed!");
    try {
        while (!feof(pipe)) {
            if (fgets(buffer, 128, pipe) != NULL)
                result += buffer;
        }
    } catch (...) {
        pclose(pipe);
        throw;
    }
    pclose(pipe);
    return result;
}

// Reads NN weights from files
shark::blas::vector<double> read_weights(){
	std::string line;
	shark::blas::vector<double> v;
	for(int i=0;i<nn_layers;i++){
		std::ifstream fin("weights/"+std::to_string(i)+".w");
		while(getline(fin,line)){
		    v.push_back(atof(line.c_str()));
		}
	}
	return v;
}

// Save NN weights to files
void write_weights(shark::blas::vector<double> v,std::string filename){
	std::ofstream file;
	file.open(filename, std::ofstream::trunc);
	for(int i=0;i<v.size();i++){
		file<<v[i]<<"\n";
	}
}
 
// Gets an objective function value
double bbox_nn(shark::blas::vector<double> init_w){
	srand(time(0));
	int rnd=rand()%100;
	write_weights(init_w,path+"weights/args/"+(std::to_string(rnd))+".a");
	std::string res=exec(("python3 "+path+"bot.py "+std::to_string(rnd)).c_str());
	std::string tmp;
	shark::blas::vector<double> score;
	for(int i=0;i<res.length();i++)
		if(isdigit(res[i]) || res[i]=='.' || res[i]=='-')
			tmp+=res[i];
		else{
			score.push_back(atof(tmp.c_str()));
			tmp="";
		}
	std::cout<<-score[0]<<std::endl;
	return -(score[0]);
}

struct Bbox : public SingleObjectiveFunction {
        Bbox(std::size_t numberOfVariables):m_numberOfVariables(numberOfVariables) {
                 m_features |= IS_THREAD_SAFE;
        }
        std::string name(){ return "BBox"; }
        std::size_t numberOfVariables()const{
                return m_numberOfVariables;
        }
        bool hasScalableDimensionality()const{
                return true;
        }
        void setNumberOfVariables( std::size_t numberOfVariables ){
                m_numberOfVariables = numberOfVariables;
        }
        double eval(const SearchPointType &p) const {
                m_evaluationCounter++;
                return bbox_nn(p);
        }
        private:
                std::size_t m_numberOfVariables;
};

int main( int argc, char ** argv ) {
 	size_t lambda,mu;
	float initialSigma;
	shark::blas::vector<double> init_w=read_weights();
	Bbox bbnn(init_w.size());
	
	// Initial parameters
	lambda=(size_t)(4 + 3 * log(init_w.size()));
	mu=(size_t)(lambda/2);
	initialSigma=1;
	
	// CMA initialization
	CMA cma;
	cma.init(bbnn,init_w,(int)(4 + 3 * log(init_w.size())),(int)(lambda/2),initialSigma);
	
	// Optimization loop
	do {
		cma.step( bbnn ); 
	} while(true);	
}
