import os
import json

txt_file = "datasets/train/train_label.txt"
save_file = "datasets/annots/train_label.json"

with open(txt_file, 'r') as f:
    annots_info = f.readlines()

save_list = []
image_id_dict = {}
idx = 1
for annot_info in annots_info:
    annot_info = annot_info.strip()
    image_name, caption = annot_info.split("$")[::2]
    if image_id_dict.get(image_name, None):
        image_id = image_id_dict.get(image_name)
        save_list.append({"image":image_name,"image_id":image_id, "caption":caption})
    else:
        save_list.append({"image":image_name,"image_id":str(idx), "caption":caption})
        image_id_dict[image_name] = idx
        idx += 1
with open(save_file, 'w') as f:
    json.dump(save_list,f, indent=4)

    



