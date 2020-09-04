import sys
import os
import itertools
import numpy as np


#-------------------------------------------------------------------------------
#                           Collect all safe channels
#-------------------------------------------------------------------------------
ifo = sys.argv[1]
if ifo == 'L1':
    ini_file = '/home/rich.ormiston/ligo-channel-lists/O3/L1-O3-standard.ini'
elif ifo == 'H1':
    ini_file = '/home/rich.ormiston/ligo-channel-lists/O3/H1-O3-standard.ini'
else:
    sys.exit('Unknown IFO')

safe_chans = []
with open(ini_file) as f:
    lines = f.readlines()
    for line in lines:
        if 'unsafe' in line:
            pass
        elif line.startswith('\t'):
            safe_chans.append(line.strip().split(' ')[0])

channel_pairs = list(itertools.combinations(safe_chans, 2))
channel_pairs = channel_pairs[:100000]

#-------------------------------------------------------------------------------
#                            Write the .dag File
#-------------------------------------------------------------------------------
dag = '/home/rich.ormiston/refinery/condor/nonlinear_pipeline.dag'
try:
    os.system('rm {}'.format(dag))
except:
    pass

sub_path = '/home/rich.ormiston/refinery/condor/nonlinear_pipeline.sub'
f = open(dag, 'a')
for ix, pair in enumerate(channel_pairs):
    st1 = 'JOB {0} {1}\n'.format(ix+1, sub_path)
    st2 = 'VARS {0} chan1="{1}" chan2="{2}" jobNumber="{0}"\n\n'.format(ix+1, pair[0], pair[1])
    f.write(st1)
    f.write(st2)
f.close()

parent = ['PARENT', 1, 'CHILD']
parent.extend(list(range(2, len(channel_pairs)+1)))
parent = [str(x) for x in parent]
parent = ' '.join(parent)

with open(dag, 'a') as f:
    f.write('{}'.format(parent))
