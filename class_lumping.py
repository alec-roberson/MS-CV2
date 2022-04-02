import os
import shutil

# class names
with open('class-names-old.txt') as f:
    classes = f.readlines()

# new classes and class mappings
with open('class-names-D1.txt', 'r') as f:
    new_classes = [c.strip() for c in f.readlines()]

class_mapping = {
    0: 7,
    1: 8,
    2: 9,
    3: 10,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 0,
    9: 0,
    10: 1,
    11: 1,
    12: 16,
    13: 17,
    14: 18,
    15: 2,
    16: 11,
    17: 12,
    18: 13,
    19: 14,
    20: 15
}

# copying/modifying files
filenames = os.listdir('raw-data')

for fname in filenames:
    old_file = os.path.join('raw-data', fname)
    new_file = os.path.join('raw-data-D1', fname)
    # any other file
    if os.path.splitext(fname)[1] != '.txt':
        shutil.copyfile(old_file, new_file)
        continue
    
    # what to do with text files
    try:
        with open(old_file, 'r') as f:
            detections = f.readlines()
    except:
        print(fname)
    
    new_detections = []
    for d in detections:
        old_class = int(d.split()[0])
        
        new_class = class_mapping[old_class]
        
        new_d = d
        if old_class >= 10:
            new_d = str(new_class)+ new_d[2:]
        else:
            new_d = str(new_class) + new_d[1:]
        new_detections.append(new_d.strip())
    
    with open(new_file, 'w') as f:
        f.write('\n'.join(new_detections))
