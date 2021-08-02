import main
import sys
from collections import Counter

word_file = main.Load('word.txt')
word_file.append('a')
word_file.append('i')
word_file = sorted(word_file)

ini_name = input('please input a word that u wanna find:\n')


def find_anagrams(name, word_list):
    name_letter_map = Counter(name)
    anagrams = []
    for word in word_list:
        test = ''
        word_letter_map = Counter(word.lower())
        for letter in word:
            if word_letter_map[letter] <= name_letter_map[letter]:
                test += letter
        if Counter(test) == word_letter_map:
            anagrams.append(word)
    print(*anagrams, sep='\n')
    print()
    print('remaining letters = {}'.format(name))
    print("whose\'s number is: {}".format(len(name)))
    print('number of remaining (real word) anagrams = {}'.format(len(anagrams)))


def process_choice(name):
    while True:
        choice = input('\nMake a choice else Enter to start over or # to end: ')
        if choice == '':
            main()
        elif choice == '#':
            sys.exit()
        else:
            candidate = ''.join(choice.lower().split())
        left_over_list = list(name)
        for letter in candidate:  # 找到剩余的字母
            if letter in left_over_list:
                left_over_list.remove(letter)
        if len(name) - len(left_over_list) == len(candidate):
            break
        else:
            print('wont work and make another choice', file=sys.stderr)
    name = ''.join(left_over_list)
    return choice, name


def main():
    name = "".join(ini_name.lower().split())
    name = name.replace('_', '')
    limit = len(name)
    phrase = ''
    running = True

    while running:
        temp_phrase = phrase.replace(' ', '')
        if len(temp_phrase) < limit:
            print('length of anagram phrase = {}'.format(len(temp_phrase)))
            find_anagrams(name, word_file)
            print("current anagram phrase = ", end=" ")
            print(phrase, file=sys.stderr)
            choice, name = process_choice(name)
            phrase += choice + ' '
        elif len(temp_phrase) == limit:
            print('\n*****Finish*****\n')
            print("anagram of name = ", end=" ")
            print(phrase, file=sys.stderr)
            print()
            try_again = input('\n\nTry again?\n(press Enter else n to quit)\n')
            if try_again.lower() == 'n':
                running = False
                sys.exit()
            else:
                main()


if __name__ == "__main__":
    main()
