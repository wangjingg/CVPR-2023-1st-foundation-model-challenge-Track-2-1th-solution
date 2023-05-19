import json

json_file = "./result_json/infer_json.json"
txt_file = open("./scripts/test_vehicle_count.json", 'r')
save_file = open("./result_json/infer_json_opt.json", 'w')

with open(json_file, 'r') as f:
    js_f = json.load(f)
label_count = json.load(txt_file)


for idx,result in enumerate(js_f['results'][:7611]):
    txt = js_f['results'][idx]['text']
    if txt.startswith('This is a '):
        txt = txt.replace('This is a ', '', 1)
    if  txt.startswith('This is an '):
        txt = txt.replace('This is an ', '', 1)
    if txt.startswith('A '):
        txt = txt.replace('A ', '', 1)
    if txt.startswith('An '):
        txt = txt.replace('An ', '', 1)
    topk = label_count[txt]

    topk = min(topk, 10)
    if topk > 3:
        js_f['results'][idx]['image_names'] = js_f['results'][idx]['image_names'][:topk] + [js_f['results'][idx]['image_names'][topk-1]]*(10-topk)
json.dump(js_f, save_file, indent=4)


