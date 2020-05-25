import os
path = os.getcwd()
print(path)

with open(path+r'/monodepth2/splits/eigen_full/val_files.txt','r') as s:
    with open(path+r'/monodepth2/splits/1003/val_files.txt','w') as d:
        lines = s.readlines()
        for line in lines:
            if "10_03" in line:
                d.write(line)