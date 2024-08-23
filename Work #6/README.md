# Генерация фракталов средствами CUDA

## Немного истории

Отдельный раздел компьютерной графики - фрактальная геометрия. При помощи явленя самоподобия можно получать достаточно причудливые узоры. Здесь отдельно стоит выделить фракталы комплексной плоскости. 
На весьма простом преобразовании вида

$$ z\to z^{2}+c $$

можно получить множества Мандельброта и Жюлиа. 

Впервые множество Мандельброта было описано в 1905 году Пьером Фату (фр. Pierre Fatou), французским математиком, работавшим в области аналитической динамики комплексных чисел. Фату изучал рекурсивные процессы данного вида.
Начав с точки ` Z = 0 `  на комплексной плоскости, можно получить новые точки, последовательно применяя к ним эту формулу. Фату нашел, что границы для данного случая показывают достаточно сложное и интересное поведение. 

Существует бесконечное множество таких преобразований — своё для каждого значения c. В те времена компьютеров ещё не было, и Фату, конечно, не мог построить орбиты всех точек плоскости, ему приходилось всё делать вручную. 
Основываясь на своих расчётах, он доказал, что орбита точки, лежащей на расстоянии больше 2 от начала координат, всегда уходит в бесконечность.
Фату никогда не видел изображений, которые мы сейчас знаем как изображения множества Мандельброта, потому что необходимое количество вычислений невозможно провести вручную. 

Профессор Бенуа Мандельброт был первым, кто использовал компьютер для визуализации множества.
Фракталы были описаны Мандельбротом в 1975 году в его книге *«Les Objets Fractals: Forme, Hasard et Dimension» («Фрактальные объекты: форма, случайность и размерность»)*. 
В этой книге Мандельброт впервые использовал термин «фрактал» для обозначения математического феномена, демонстрирующего столь непредсказуемое и удивительное поведение. 

Эти феномены рождались при использовании рекурсивного алгоритма для получения какой-либо кривой или множества. Множество Мандельброта — один из таких феноменов, названный по имени своего исследователя.
В 1978 году фрактал был определён и нарисован Робертом У. Бруксом и Питером Мательским как часть исследования групп Клейна. 1 марта 1980 года Бенуа Мандельброт первым увидел визуализации множества. 
Математическое исследование множества Мандельброта началось с работы математиков Адриана Дуади (Adrien Douady) и Джона Х. Хаббарда (John H. Hubbard), которые установили многие из его фундаментальных свойств.

## Проверка точки на принадлежность множеству

Множество лежит в комплексной плоскости, и включает в себя бесконечное число точек, большая часть которых сосредоточена в интервале от -1 до 1 по оси Imaginary и в интервале от -2 до 2 по опи Real. 
Для удобства визуализации мы буедм использовать декартову систему координат. Мы не будем использовать встроенные функции для работы с комплексными числами, а опишем все необходимые преобразования вручную.

Чтобы проверить, принадлежит ли точка с координатами (c_rl, c_im) ко множеству Мандельброта нам необходимо определить, превысит ли модуль комплексного числа Z значнеие 2 при применение данного преобразования или нет. 
Делать такое преобразование бесконечное число раз мы не будем, а ограчимся 256 итерациями. Так будет и нагляднее визуализация. 

``` math
Z_0 = 0
```
``` math
Z_1 = Z_0^2 + C 
```
``` math
...
```
``` math
Z_i = Z_{i-1}^2 + C 
```

Условием выхода из цикла будем счиатать случай, когда модуль уже на текущей итерации превысил 2.

``` math
|Z_{i}| > 2
```

Напишем небольшой код для определения принадлежности числа ко множеству:

``` cuda
int checkPoint(double c_rl, double c_im) {
  // Устанавливаем Z = 0
  double z_rl = 0;
  double z_im = 0;
  int i = 0;

  for (i = 0; i < 256; i++) {
    double z_rl_2 = z_rl * z_rl - z_im * z_im;
    double z_im_2 = 2 * z_im * z_rl;
    z_rl = z_rl_2;
    z_im = z_im_2;
    z_rl = z_rl + c_rl;
    z_im = z_im + c_im;

    // Если модуль уже превысил 2, то дальше нет смысла считать
    if (z_rl * z_rl + z_im * z_im > 4) return i;
  }
  return 255;
}
```

Код получился достаточно небольшой. Однако он содержит некоторые "неудобства", которые мешают использовать его в качестве kennel в CUDA.
1. Передаются координаты в дробном формате. CUDA поддерживает только целые положительные числа при определении индексов потоков
2. Результатом работы должно быть изображение, которое можно будет наглядно оценить
3. Ответ должен не возвращаться, а записываться в результирующий массив

Чтобы предупредить возможные проблемы, давайте возьмём код из предыдущего проекта и уберем все лишнее:

``` cuda
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <chrono>

using namespace cv;

__global__ void RGB2GRAY_GPU(uchar *gp_clr, uchar *gp_gray, int rows, int cols) {
    // Получаем координаты на основании ID блока и треда
	int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	// Определяем индекс пикселя по такой формуле
	int idx = y * cols + x;
	
	// Проверяем выходы за границу
    if (x > cols || y > rows)
        return;
	
	// Считаем сумму
	int sum = (gp_clr[idx*3+0]+gp_clr[idx*3+1]+gp_clr[idx*3+2]) / 3;
	
	// Записываем в реузльтирующий
	gp_gray[idx] =  sum;
    return;
} 

int main(int argc, char** argv ) {	
  // Два холста - исходный и результирующий. Первый - цветное изображение, второй - одноканальное пустое изображение
  //Mat color_img = imread("test.jpg", IMREAD_COLOR );
	//int cols = color_img.cols;
	//int rows = color_img.rows;	
  int cols = 600;
  int rows = 600;

	Mat gray_img = Mat::zeros(rows, cols,CV_8UC1);

	// Берем указатели на данные из изображения
	//unsigned char* p_clr = color_img.ptr<unsigned char>();
	unsigned char* p_gray = gray_img.ptr<unsigned char>();
	
	std::chrono::steady_clock::time_point begin, end;
	begin = std::chrono::steady_clock::now();
	
	//unsigned char *gp_clr;
  unsigned char *gp_gray;
	
	//cudaMalloc(&gp_clr, sizeof(unsigned char) * rows * cols * 3);
  cudaMalloc(&gp_gray, sizeof(unsigned char) * rows * cols);
	
	int THREAD_DIM = 32;
	dim3 gridSize (ceil(cols / (float)THREAD_DIM), ceil(rows / (float)THREAD_DIM));
  dim3 blockSize (THREAD_DIM, THREAD_DIM);

  //cudaMemcpy(gp_clr, p_clr, sizeof(unsigned char) * rows * cols * 3, cudaMemcpyHostToDevice);
    
	RGB2GRAY_GPU<<< gridSize, blockSize>>>(gp_clr, gp_gray, rows, cols);
	
	cudaMemcpy(p_gray, gp_gray, sizeof(unsigned char) * rows * cols, cudaMemcpyDeviceToHost);

	end = std::chrono::steady_clock::now();
	std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[µs]" << std::endl;
	begin = std::chrono::steady_clock::now();
	
	// Сохраняем изображение
  imwrite("uu.png", gray_img);
  return 0;
}
```

В результате получим следующую заготовку:

``` cuda
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <random>
#include <cstdlib>
#include <cstdio>
#include <chrono>

using namespace cv;

__global__ void RGB2GRAY_GPU(uchar *gp_gray, int rows, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	int idx = y * cols + x;
	
	// Проверяем выходы за границу
  if (x > cols || y > rows)
        return;
	
	// Что-то делаем ...
	
	// Записываем результат
	gp_gray[idx] = 0;
  return;
} 

int main(int argc, char** argv ) {		
	int cols = 600;
	int rows = 600;
	Mat gray_img = Mat::zeros(rows, cols,CV_8UC1);
	unsigned char* p_gray = gray_img.ptr<unsigned char>();
	unsigned char *gp_gray;
	cudaMalloc(&gp_gray, sizeof(unsigned char) * rows * cols);
	
	int THREAD_DIM = 32;
	dim3 gridSize (ceil(cols / (float)THREAD_DIM), ceil(rows / (float)THREAD_DIM));
  dim3 blockSize (THREAD_DIM, THREAD_DIM);

	RGB2GRAY_GPU<<< gridSize, blockSize>>>(gp_clr, gp_gray, rows, cols);
	
	cudaMemcpy(p_gray, gp_gray, sizeof(unsigned char) * rows * cols, cudaMemcpyDeviceToHost);

	imwrite("fractal.png", gray_img);
	return 0;
}
```

Здесь осталось добавить наши вычисления в kernel и немного модернизровать его вызов.

``` cuda
__global__ void GetColor(uchar *result, int cols, int rows, double scale, double p_x, double p_y) {

  ...

  double c_rl = (x-cols/2) / scale + p_x;
  double c_im = (y-rows/2) / scale + p_y;
  double z_rl = 0;
  double z_im = 0;
  int i = 0;

  for (i = 0; i < 255; i++) {
    double z_rl_2 = z_rl * z_rl - z_im * z_im;
    double z_im_2 = 2 * z_im * z_rl;
    z_rl = z_rl_2;
    z_im = z_im_2;
    z_rl = z_rl + c_rl;
    z_im = z_im + c_im;

    if (z_rl * z_rl + z_im * z_im > 4)
      break;
  }
  result[idx] = i;

...
```

В целом мы повторили код выше. Однако, здесь есть некоторые дополнения, о которых стоит сказать. Были несколько модифицированны методы определения точки C. Приведенный выше подход позволил использовать целые числа в качестве индекса, а также поставить по центру интересующую нас точку и добавить варьируемый коэффициент масштабирования. В этоге наш фрактал можно отдалять и приближать при изменении данного значения. 

Задание: не забудьте модернизировать вызов kernel. В качестве точки приближения выберите (0, 0), а масштаб - 200. 


