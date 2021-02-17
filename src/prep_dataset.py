# %%
import os
import glob
from shutil import copyfile

labeler = 'Degenhardt'
labeldir = './labels2'
classnames_final = ['Neurogen', 'Strukturell']

labels = os.listdir(labeldir)

label = labels[0]
classname = label.split('_')[1]
label.strip(labeler).strip(f'_{classname}_').strip('.txt')

# %%

for label in labels:
    classname = label.split('_')[1]
    rawfilename = label.strip(labeler).strip(f'_{classname}_').strip('.txt')

    source = f'./{classname}_raw/{rawfilename}'
    destination = f'./{classname}/{rawfilename}'
    copyfile(source, destination)

# %%
