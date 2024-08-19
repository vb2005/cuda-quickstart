# Работа №1. Получение данных об устройствах с поддержкой CUDA

Рассмотрим несколько функций, которые предоставляет библиотека cuda_runtime.h

Начнём проект с создания пустого файла example1.cu. Чтобы не копировать код, Вы можете воспользоваться Git-репозиторием.

#include "cuda_runtime.h"
#include <cstdlib>
#include <cstdio>
#include <iostream>

int main()
{
	std::cout << "Hello CUDA!" << std::endl;
	return 0;
}


Теперь файл необходимо собрать и запустить. Для сборки мы будем использовать компилятор nvcc. Он собирает исполняемый файл с поддержкой всех необходимых библиотек. Следующие команды мы будем выполнять всегда, для компиляции и запуска нового проекта. 

Для Linux (Google Colab):
nvcc -I/usr/local/cuda/include -L/usr/local/cuda/lib example1.cu -o test
/content/test

 
Для Windows:
1. Убедитесь, что установлен Visual Studio 2017+ с поддержкой собрки приложений C/C++. Также убедитесь, что установлены драйвера и NVidia CUDA Toolkit.
2. Зайдите в меню Пуск - Visual Stuido - x64 Native Tools Command Prompt for VS
3. Откроется командная строка разработки C/C++. Перейдите в каталог C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.3\bin
4. Запустите сборку командной:
nvcc <Путь к исходному коду> -o <Путь к сохраняемому exe-файлу>
5. Запустите пример и убедитесь в корректности его работы


Теперь добавим несколько операторов для того, чтобы определить установленные видеоускорители

int count;
cudaGetDeviceCount(&count);
std::cout << "Доступно для использования " << count << " устройств с поддержкой CUDA" << std::endl;

Функция cudaGetDeviceCount передаёт в аргумент сведения об установленных видеоускорителях. Добавьте эти строки кода и проверьте, поддерживает ли Ваша видеокарта неграфические вычисления CUDA

# Получение сведений об устройстве CUDA

Далее мы будем работать с первым найденным устройством. Получим все его характеристики и выведем их на экран.

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

Запомните значение Max Threads per Block. Это важное для нас значение будет определять то, сколько параллельно потоков мы сможем одновременно выполнить. 





























