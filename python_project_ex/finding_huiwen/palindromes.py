import main

word = main.Load('word.txt')
word = set(word)
'''
for i in word:
    print(i)
    break

tar_word = []
for i in word:
    if i[:] == i[::-1] and len(i) > 1:
        tar_word.append(i)
print(len(tar_word))
print(*tar_word, sep='\n')
'''
def find_palingrams():
    pali = []
    for i in word:
        end = len(i)
        rev = i[::-1]
        if end > 1:
            for j in range(end):
                if i[j:] == rev[:end-j] and rev[end-j:] in word:
                    pali.append((i, rev[end - j:]))
                if i[:j] == rev[end-j:] and rev[:end-j] in word:
                    pali.append((rev[:end - j], i))
    return pali

palingrames = find_palingrams()
pali_sorted = sorted(palingrames)
print(len(pali_sorted))
for fir, sec in pali_sorted:
    print('{} {}'.format(fir, sec))
