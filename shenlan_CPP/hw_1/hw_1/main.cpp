#include <iostream>
#include <cstdlib>
#include <time.h>

using namespace std;
int rand_alg(int mode)//���ѡ���Ӽ��˳��е�һ������
{
	char alg;
	srand(int(time(0)));
	if (mode == 1)
	{
		int m = rand() % 10;
		if (m < 5)
		{
			alg = '+';
		}
		else
		{
			alg = '-';
		}
	};
	if (mode == 2)
	{
		int m = rand() % 20;
		if (m < 10)
		{
			if (m < 5)
			{
				alg = '+';
			}
			else
			{
				alg = '-';
			}
		}
		else
		{
			if (m < 15)
			{
				alg = '/';
			}
			else
			{
				alg = '*';
			}
		}
		return alg;
	}

}
int caculate(char alg, int a, int b)//��������ѡ���������µ���ȷֵ
{
	int right = 0;
	if (alg == '+')
	{
		right = a + b;
	}
	else if (alg == '-')
	{
		right = a - b;
	}
	else if (alg == '*')
	{
		right = a * b;
	}
	else if (alg == '/')
	{
		right = a / b;
		cout << "ps��Just write the integer part haha " << endl;
	}
	return right;
}
int main()
{
	srand(int(time(0)));
	int num = 3; // �ܹ�ѭ������
	clock_t start, end;
	int range = 10; //ѡ��0������ �����Χ������
	int mode = 1; //ѡ�����ѻ��Ǽ�ģʽ
	double corr = 0.0; //���ڼ������յ÷֣��ٷ��ƣ�
	double time_all = 0.0; //���ڼ��������ʱ����λΪs��
	int round = 0; //���ڼ������ٺ�ʱ�ִ�
	int round_min = 0;
	double time_min = 1000000;
	cout << "how many questions u want? : \n" << endl;
	cin >> num;
	cout << "which range can u accept ? from 0 to ?: \n" << endl;
	cin >> range;
	cout << "which mode do u want ? input 1 for only +&-, or 2 for *&/ extra :  \n" << endl;
	cin >> mode;

	for (int x = 0; x < num; x += 1)
	{
		int a = rand()%range;
		int b = rand()%range;
		char alg = rand_alg(mode);
		int c;
		cout << " so what is " << a << alg << b << '?' << endl;
		int right = caculate(alg, a, b);
		start = clock();
		cin >> c;
		end = clock();
			if (c!=right)
			{
				if (x != (num - 1))
				{
					cout << " u r wrong�� and the right answer is  " << (a + b) << endl;
				}
				else
				{
					corr = (corr / double(num)) * 100;
					cout << " u r wrong ,end of the prc and your score is(up to 100): \n " << corr << endl;
				}
			}
			else
			{
				if (x != (num -1))
				{
					cout << " u r right��  \n  " << endl;
					corr += 1;
				}
				else
				{
					corr += 1;
					corr = (corr / double(num)) * 100;
					cout << "u r right! end of the prc and your score is(up to 100): \n " << corr << endl;
				}
			}
			double time = end - start;
			time_all += time;
			round += 1;
			if (time < time_min)
			{
				time_min = time;
				round_min = round;
			}
			cout << "the time u cost this round is:   " << time / CLOCKS_PER_SEC << "s \n" << endl;
	};
	double time_avg = time_all / double(num);
	cout << "the time u cost for average is:   " << time_avg / CLOCKS_PER_SEC << "s \n" << endl;
	cout << "the time u cost at least is:   " << time_min / CLOCKS_PER_SEC << "s  in round " << round_min << endl;
}
