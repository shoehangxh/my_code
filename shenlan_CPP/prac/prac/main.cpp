#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;
int main()
{
	int right = 0;
	char alg = '+';
	int a = 9;
	int b = 3;
	/*
	char c = '+';
	char d = '-';
	char e = '*';
	char f = '/';
	const char *aim = &alg;
	const char *plus = &c;
	const char *minus = &d;
	const char *multi = &e;
	const char *division = &f;*/
	if (alg == '+')
	{
		right = a + b;
		cout << right << endl;
	}
	/*
	else if(strcmp(aim, minus) == 0)
	{
		right = a - b;
		cout << right << endl;
	}
	else if (strcmp(aim, multi) == 0)
	{
		right = a * b;
		cout << right << endl;
	}
	else if (strcmp(aim, division) == 0)
	{
		right = a / b;
		cout << right << endl;
	}
	*/
	/*cout << right << endl;
	cout << *aim << endl;
	cout << *plus << endl;
	cout << *minus<< endl;
	cout << *multi << endl;
	cout << *division << endl;*/

}
