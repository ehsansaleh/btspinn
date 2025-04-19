import os
from textwrap import dedent

cwd_list = os.getcwd().split(os.sep)
projname = 'btspinn' if ('code_bspinn' not in cwd_list) else 'code_bspinn'
assert cwd_list.count(projname) == 1, dedent(f"""
    The project name "{projname}" must appear exactly
    once in the current working directory:
        >> os.getcwd() == {os.getcwd()}""")
PROJPATH = os.sep.join(cwd_list[:cwd_list.index(projname) + 1])

configs_dir = f'{PROJPATH}/configs'
results_dir = f'{PROJPATH}/results'
storage_dir = f'{PROJPATH}/storage'
summary_dir = f'{PROJPATH}/summary'
source_dir = f'{PROJPATH}/bspinn'

# The key specs used for summarization
keyspecs = [('hp',         'last'),
            ('stat',       'mean'),
            ('etc',        'last'),
            ('mon',        'mean'),
            ('mdl',        'last'),
            ('trg',        'last'),
            ('var/eval/*', 'last')]

nullstr = '-'
