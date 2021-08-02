import sys
import random
print("welcome to the Psych 'Sideknick Name Picker'\n")
print('a name just like Sean would pick for Gus:\n\n')


first = ('啊这', '这是啥', '善良', '感恩', '的心呐', '网抑云', '王者荣亚')
last = ('pyhton', 'zhaojunjie', 'luoyingwu', 'gaoxiang', 'hhh', 'zju', 'kaikeba')

while True:
    firstname = random.choice(first)
    lastname = random.choice(last)
    print('\n\n')
    print('{} {}'.format(firstname, lastname), file=sys.stderr)
    # 把文字设置为红色
    print('\n\n')
    tryagain = input('\n\nTry Again?, (Press Enter else n to quit)\n')
    # .lower()将做所有文字转化为小写
    if tryagain.lower() == 'n':
        break

input('\nPress Enter to exit.')
