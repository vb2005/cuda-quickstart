#include "cuda_runtime.h"
#include <cstdlib>
#include <cstdio>
#include <iostream>

int main()
{
	// Получаем количество подключенных устройств
	int count;
	cudaGetDeviceCount(&count);
	std::cout << "Доступно для использования " << count << " устройств с поддержкой CUDA" << std::endl;
	
	// Использовать будем первое найденное устройство
	cudaDeviceProp prop;
	if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return -1;

	std::cout << "Major version of first CUDA device: " << prop.major << std::endl;
	std::cout << "Minor version of first CUDA device: " << prop.minor << std::endl;
	std::cout << "Is ECC Enabled: "                     << prop.ECCEnabled << std::endl;
	std::cout << "Async Engine Count: "                 << prop.asyncEngineCount << std::endl;
	std::cout << "Can map host memory: "                << prop.canMapHostMemory << std::endl;
	std::cout << "clock rate: "                         << prop.clockRate << std::endl;
	std::cout << "Concurrent Kernels: "                 << prop.concurrentKernels << std::endl;
	std::cout << "Global L1 Cache supported: "          << prop.globalL1CacheSupported << std::endl;
	std::cout << "Max Threads Dimension: "              << prop.maxThreadsDim << std::endl;
	std::cout << "Max Threads per Block: "              << prop.maxThreadsPerBlock << std::endl;
	std::cout << "Max Threads per MultiProcessor: "     << prop.maxThreadsPerMultiProcessor << std::endl;
	std::cout << "MultiProcessorCount: "                << prop.multiProcessorCount << std::endl;
	std::cout << "Device Name: "                        << prop.name << std::endl;
	std::cout << "Total Global Memory: "                << prop.totalGlobalMem << std::endl;
	std::cout << std::endl;
	
	std::cout << "Hello CUDA!" << std::endl;
	return 0;
}
