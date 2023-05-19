import os
from tqdm import tqdm
import json
import random
with open('/mnt/wj/paddle_image_retrieval_data/test/test_text.txt', 'r') as f:
    lines = f.readlines()

save_txt = open('./scripts/test_vehicle_count.json', 'w')


txt_dict = {}
for line in tqdm(lines[:7611]):
    wenben = line.strip()

    if wenben.startswith('This is a '):
        label = wenben.replace('This is a ', '', 1)
    if  wenben.startswith('This is an '):
        label = wenben.replace('This is an ', '', 1)
    if wenben.startswith('A '):
        label = wenben.replace('A ', '', 1)
    if wenben.startswith('An '):
        label = wenben.replace('An ', '', 1)
        
    if txt_dict.get(label, None):
        txt_dict[label] += 1
    else:
        txt_dict[label] = 1

txt_str = json.dumps(txt_dict, indent=4)
save_txt.write(txt_str)


