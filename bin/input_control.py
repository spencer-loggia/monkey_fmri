from typing import List

import os


def dir_input(msg: str):
    path = "/not_a_path"
    while not os.path.exists(path):
        path = input(msg)
    return path


def bool_input(msg: str):
    tkn = ''
    while tkn not in ['y', 'n']:
        tkn = input(msg + ' (y / n) ').strip().lower()
    return tkn == 'y'


def int_list_input(msg: str):
    grammatical = False
    nums = []
    while not grammatical:
        sent = input(msg)
        tkns = sent.split()
        try:
            nums = [int(t) for t in tkns]
        except ValueError:
            try:
                nums = [float(t) for t in tkns]
            except ValueError:
                continue
        break
    return nums


def tuple_list_input(msg: str, cast=int):
    grammatical = False
    tuples = []
    while not grammatical:
        sent = input(msg + ' <format: (item_11, ..., item_1n), ..., (item_n1, ..., item_nn)> : ')
        sent = ''.join(sent.split()) # remove whitespace
        sent = sent[1:-1]
        tkns = sent.split('),(')
        for tkn in tkns:
            try:
                tup = eval('[' + tkn + ']')
                if type(tup) is not list:
                    raise ValueError
            except (ValueError, SyntaxError):
                print("Failed to evaluate term token " + tkn + " to list.")
                continue
            try:
                tup = [cast(t) for t in tup]
            except ValueError:
                print('failed to cast some values to ' + str(cast))
                continue
            tuples.append(tuple(tup))
        break
    return tuples


def str_list_input(msg: str):
    sent = input(msg)
    tkns = sent.split()
    return tkns


def select_option_input(option_desc: List[str]):
    print('********************************')
    for i, option in enumerate(option_desc):
        print(i + 1, ':', option_desc[i])
    good_select = False
    help_req = False
    while not good_select:
        try:
            choice = input('Enter the number for the desired operation: ')
            if '-h' in choice:
                choice = int(choice.split()[0])
                help_req = True
            else:
                choice = int(choice)
        except ValueError:
            continue
        if 0 < choice <= len(option_desc):
            if help_req:
                return choice - 1, True
            return choice - 1
