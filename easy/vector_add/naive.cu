#include <cuda_runtime.h>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <errno.h>
#include <vector>
#include <stdexcept>
#include <new>
#include <cmath>

#define ERROR_LOG(fmt,...) \
	std::fprintf(stderr,"\n [ERROR] FILE:%s LINE:%d DETAIL:" fmt, \
			__FILE__,__LINE__,##__VA_ARGS__)


#define CUDA_CHECK(call) do{ \
	cudaError_t err = (call); \
	if(err != cudaSuccess){ \
		ERROR_LOG("%s: %s",#call,cudaGetErrorString(err));\
		exit(EXIT_FAILURE); \
	} \
}while(0)





__global__ void add(const float* a,const float* b,float* c,int n){
        int i=blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n) c[i] = a[i]+b[i];
}


void genRandom(std::vector<float>& h_a,std::vector<float>& h_b,int n){
	std::mt19937 gen(42);   //引擎
	std::uniform_real_distribution<float> dist(-1.0f,1.0f);// 分布
	
	for(int i=0;i<n;++i){
		h_a[i] = dist(gen);
		h_b[i] = dist(gen);
	}
	
}

inline void CudaAdd(int n){
	//生成host侧
	std::vector<float> h_a(n),h_b(n);
	genRandom(h_a,h_b,n);
	std::vector<float> h_c(n);
	
	//Device侧
	float *d_a,*d_b,*d_c;
	constexpr int threads=256;
	const int blocks=(n+threads-1)/threads;
	size_t bytes=n*sizeof(float);
	CUDA_CHECK(cudaMalloc((void**)&d_c,bytes));
	CUDA_CHECK(cudaMalloc((void**)&d_a,bytes));
	CUDA_CHECK(cudaMalloc((void**)&d_b,bytes));
	CUDA_CHECK(cudaMemcpy(d_a,h_a.data(),bytes,cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(d_b,h_b.data(),bytes,cudaMemcpyHostToDevice));	
	
	cudaEvent_t start,stop;
	CUDA_CHECK(cudaEventCreate(&start));
	CUDA_CHECK(cudaEventCreate(&stop));
	
	//warm-up
	add<<<blocks,threads>>>(d_a,d_b,d_c,n);
	CUDA_CHECK(cudaDeviceSynchronize());
	

	CUDA_CHECK(cudaEventRecord(start));
	//Kernel launch
	add<<<blocks,threads>>>(d_a,d_b,d_c,n);
	CUDA_CHECK(cudaEventRecord(stop));
	
	// kernel result and synchronize stream
	CUDA_CHECK(cudaGetLastError());
	CUDA_CHECK(cudaEventSynchronize(stop));
	
	float ms=0.0f;
	CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
	
	CUDA_CHECK(cudaEventDestroy(start));
	CUDA_CHECK(cudaEventDestroy(stop));
	
	// printf kernel run time
	printf("\n [ADD_RUNTIME]:%.4f ms",ms);
	//copy from kernel
	CUDA_CHECK(cudaMemcpy(h_c.data(),d_c,bytes,cudaMemcpyDeviceToHost));
	
	//to check the error by compare host and kernel result
	
	int failedCount=0;
	for(int i=0;i<n;i++){
		float predictValue=h_a[i]+h_b[i];
		if(fabs(predictValue-h_c[i])>1e-2){
			failedCount++;	
			#ifdef DEBUG
				printf("\n[FAILED] :Number: %d, excpected: %.2f,real: %.2f ",i,predictValue,h_c[i]);

			#endif 	
		}
		else{	
			#ifdef DEBUG
				printf("\n[PASS] :Number: %d,value: %.2f",i,predictValue);
			#endif
		}
	}
	printf("\n[Sumary]: %d",failedCount);

	CUDA_CHECK(cudaFree(d_a));
	CUDA_CHECK(cudaFree(d_b));
	CUDA_CHECK(cudaFree(d_c));

}


int main(){
	std::vector<int> N{0,1,31,255,256,257,1000,4096,1<<20};
	for(size_t i=0;i<N.size();++i){
		if(N[i]>0)
		   CudaAdd(N[i]);
	}
	return 0;
}




