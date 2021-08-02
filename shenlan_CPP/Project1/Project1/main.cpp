#include <iostream>
#include <limits>

struct STR
{
	char b;
	int x;
};

extern int g_x;

int main()
{
	// 8000~8003 not 7999~8002 (actually, 8000~8064 for a time in cash)
	int x = 10;
	unsigned short a = 10;
	unsigned int b = -1;
	char y; //ASCII UTF-8
	wchar_t z; //unicode or char16_t,char32_t which is more common
	std::cout << x << std::endl;
	std::cout << alignof(wchar_t) << std::endl;
	std::cout << sizeof(char) << std::endl;
	std::cout << sizeof(STR) << std::endl; //not 1+4 but 8
	std::cout << b << std::endl; //4294967295
}