#include <iostream>

void fun(const char* pInfo, int /* pValue */)
	std::cout << pInfo << std::endl;
	return;
}


int main(int argc, char* argv[])
{
	fun("Hello world", 0);
	fun("Hello china", 1);
	return 0; 
}