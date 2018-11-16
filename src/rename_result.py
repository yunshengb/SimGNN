#!/usr/bin/env python3
from utils import get_result_path, exec, prompt
from os import walk
from os.path import join

f = '/aids10k_small/'
source = 'aids10k'
target = 'aids10k_small'


def rename():
    rp = get_result_path()
    for dirpath, dirs, files in walk('{}/{}'.format(rp, f)):
        for bfn in files:
            if target in bfn:
                continue
            dest_bfn = bfn.replace(source, target)
            t = prompt('Rename {} to {}? [y/n]'.format(bfn, dest_bfn), ['y', 'n'])
            if t == 'y':
                exec('mv {} {}'.format(join(dirpath, bfn), join(dirpath, dest_bfn)))
            elif t == 'n':
                print('Skip')
            else:
                assert (False)
    print('Done')


rename()
