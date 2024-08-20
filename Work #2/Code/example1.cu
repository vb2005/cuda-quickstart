#include <cstdio>

__global__
void first_kernel()
{
  printf("I'm %i thread in %i block. BlockSize is %i \n", threadIdx.x,  blockIdx.x, blockDim.x);
  __syncthreads();
  printf("Second part \n");
}

int main(){
  first_kernel<<<3, 5>>>();
  system("pause");
}