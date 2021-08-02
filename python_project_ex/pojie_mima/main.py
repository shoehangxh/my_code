list_of_lists = [['16', '12', '8', '4', '0'], ['1', '5', '9', '13','17'],
                 ['18', '14', '10', '6', '2'],
                 ['3', '7', '11', '15', '19']]
ciphertext = '16 12 8 4 0 1 5 9 13 17 18 14 10 6 2 3 7 11 15 19'
cipherlist = list(ciphertext.split())

COLS = 4
ROWS = 5
key = '-1  2  -3  4'
translation_matrix = [None] * COLS
plaintext = ''
start = 0
stop = ROWS

key_int = [int(i) for i in key.split()]
for k in key_int:
    if k<0:
        col_items = cipherlist[start:stop]
    else:
        col_items = list(reversed(cipherlist[start:stop]))
    translation_matrix[abs(k) -1] = col_items
    start += ROWS
    stop += ROWS
print(*translation_matrix, sep = '\n')

for i in range(ROWS):
    for col_items in translation_matrix:
        word = str(col_items[-1])
        del col_items[-1]
        plaintext += word + ' '
print(plaintext)
