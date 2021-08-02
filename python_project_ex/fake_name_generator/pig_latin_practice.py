def main():
    word = input('enter a word that you need to change:\n')
    word_list = list(word.lower())
    # print(word_list[0])
    if word_list[0] == 'a' or word_list[0] == 'e' or word_list[0] == 'i' or word_list[0] == 'o' or word_list[0] == 'u':
        word_list.append('way')
    else:
        word_list.append(word_list[0])
        word_list = word_list[1:]
        word_list.append('ay')
    print("".join(word_list))

if __name__  ==  '__main__':
    main()