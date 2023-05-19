# !/usr/bin/env python3
import json
import numpy as np
from collections import OrderedDict
import os 
import sys
import torch
from PIL import Image
import open_clip
import argparse

def get_images_texts(images_root, texts_root, transform_ops, test=False):
    images = []
    texts = []

    f = open(texts_root, 'r')
    for line in f.readlines():
        text = line.strip()
        texts.append(text)

    idx_dic = {}
    cnts = 0
    images_names = os.listdir(images_root)
    for image_name in sorted(images_names):
        # img = read_image(os.path.join(images_root, image_name), "RGB")
        # img = transform_ops(img)
        img = Image.open(os.path.join(images_root, image_name))
        img = transform_ops(img).unsqueeze(0)#.to(device)
        images.append(img)
        idx_dic[cnts] = image_name
        cnts += 1

    if test:
        images = images[:20]
        texts = texts[:20]

    return images, texts, idx_dic


def get_similarity(images, texts):

    # tokenizer = SimpleTokenizer()
    n = len(images)
    stage = 1
    image_features_list = []
    text_features_list = []
    for i in range(0, n, stage):
        if i % 5000 == 0:
            print(f'{i}/{n}')
        with torch.no_grad():
            image = images[i].to(device) #preprocess(images[0]).unsqueeze(0)
            text = tokenizer(texts[i]).to(device)
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features_list.append(image_features)
            
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_features_list.append(text_features)
                       
    image_features = torch.cat(image_features_list, axis=0)
    text_features = torch.cat(text_features_list, axis=0)

    print('image_features.shape', image_features.shape)
    print('text_features.shape', text_features.shape)
    similarity = torch.matmul(text_features, image_features.t()).cpu().numpy()
    
    return similarity


def infer(similarity, idx_dic, texts, task_name="infer_result"):
    np.save(task_name + ".npy", similarity)
    similarity_argsort = np.argsort(-similarity, axis=1)
    print('similarity_argsort.shape', similarity_argsort.shape)

    topk = 10
    result_list = []
    for i in range (len(similarity_argsort)):
        dic = {'text': texts[i], 'image_names': []}
        for j in range(topk):
            dic['image_names'].append(idx_dic[similarity_argsort[i,j]])
        result_list.append(dic)
    with open(task_name + '.json', 'w') as f:
        f.write(json.dumps({'results': result_list}, indent=4))


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained-path",
        type=str,
        default=None,
        help="Path to weights",
    )
    parser.add_argument(
        "--images-root",
        type=str,
        default="/home/aistudio/data/data218656/test/test_images_padding/",
    )
    parser.add_argument(
        "--texts-root",
        type=str,
        default="/home/aistudio/data/data218656/test/test_text.txt",
        help="Path to file(s) with test data",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="ViT-H-14",
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        default=None,
        help="File save name",
    )

    args = parser.parse_args(args)

    return args


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])

    device = torch.device("cuda")
    # pretrained = './pretrained/vitbase_clip.pdparams'
    pretrained_path = args.pretrained_path
    # images_root = '/media/fs/samsungSSD/cvpr2023/data_set/test/test_images/'
    images_root = args.images_root
    texts_root = args.texts_root
    model_name = args.model_name
    save_name = args.save_name

    # assert "padding" in images_root, "images must padding"

    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained_path, device=device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    # context_length = model
    # vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    # print("Context length:", context_length)
    # print("Vocab size:", vocab_size)

    images, texts, idx_dic = get_images_texts(images_root, texts_root, preprocess)
    similarity = get_similarity(images, texts)

    infer(similarity, idx_dic, texts, task_name=save_name)
    print("{} is done save file is {}.json and {}.npy".format(model_name, save_name, save_name))
