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
