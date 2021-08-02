import main

word = main.Load('word.txt')
dy = input('please input a word that u wanna find:\n')
dy = sorted(list(dy.lower()))
ny = len(dy)
list_ = []
for i in word:
    if len(i) == ny:
        dx = sorted(list(i))
        if dx == dy:
            print('here u are: {}!\n'.format(i))
            list_.append(i)
        else:
            continue
    else:
        continue
print(list_)