#include <iostream>
#include <typeinfo>
#include <vector>

extern int array[4];
int main()
{
	int a[3] = { 1, 2, 3 }; //编译期确定大小
	int x2[3][4];
	std::vector<int> x;
	std::vector<float> y;
	/*
	auto ptr = a;
	auto ptr2 = a + 2;
	std::cout << ptr2 - ptr  << std::endl;
	std::cout <<  *ptr << std::endl;
	std::cout << array[1] << std::endl;*/
	size_t index = 0;
	auto ptr = std::cbegin(a);
	while (ptr != std::cend(a))
	{
		std::cout << *ptr << std::endl;
		ptr += 1;
	}
	for (int x : a)
	{
		std::cout << x << std::endl;
	}
	/*while (index <= std::size(a))
	{
		std::cout << a[index] << std::endl;
		index += 1;
	}*/
	system("pause"); 
}