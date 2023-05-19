'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
'''

import os
import numpy as np
import torch
import torch.nn.functional as F
from data.utils import padding_resize
from tqdm import tqdm
from models.blip_retrieval import blip_retrieval
import utils
from torchvision import transforms
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

def get_transforms():
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_test = transforms.Compose([
        # transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])
    return transform_test


def get_images_texts(images_root, texts_root, transform_ops):
    images = []
    texts = []
    N = 1000000

    text_dic = {}
    dic_name_text = {}
    f = open(texts_root, 'r')
    for line in f.readlines():
        line = line.strip('')
        items = line.split('$')
        name, text, wenben = items
        text_dic[name] = text
        dic_name_text[items[0].split('.')[0]] = wenben

    idx_dic = {}
    cnts = 0
    images_names = sorted(os.listdir(images_root))
    images_names.sort()
    # images_names = images_names[:N]
    # print(images_names)
    for image_name in images_names:
        only_name = image_name.split('.')[0]
        if dic_name_text.get(only_name, -1) == -1: continue
        # if 'vehicle' in only_name: continue

        img = Image.open(os.path.join(images_root, image_name)).convert('RGB')    
        img = padding_resize(img, 224, color=(0, 0, 0))
        img = transform_ops(img)  
        images.append(img)
        texts.append(dic_name_text[only_name])
        idx_dic[cnts] = text_dic[image_name]
        cnts += 1
    images = images[:N]
    texts = texts[:N]

    return images, texts, idx_dic


@torch.no_grad()
def get_similarity(model, images, texts, device, topk = 10):
    # test
    model.eval() 
    print('Computing features for evaluation...')
    num_text = len(texts)
    batch_size = 100
    text_embeds = []
    image_embeds = []
    image_feats = []
    text_ids = []
    text_atts = []
    for i in tqdm(range(0, num_text, batch_size)):
        text = texts[i: min(num_text, i+batch_size)]
        image = images[i: min(num_text, i+batch_size)]

        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=100, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

        image = torch.cat([im.unsqueeze(dim=0) for im in image], dim=0)
        image_feat = model.visual_encoder(image.to(device))
        image_feats.append(image_feat.cpu()) 
        image_embed = model.vision_proj(image_feat[:,0,:])
        image_embed = F.normalize(image_embed,dim=-1)      
        image_embeds.append(image_embed)

    text_embeds = torch.cat(text_embeds,dim=0)
    text_ids = torch.cat(text_ids,dim=0)
    text_atts = torch.cat(text_atts,dim=0)
    text_ids[:,0] = model.tokenizer.enc_token_id

    image_feats = torch.cat(image_feats,dim=0)
    image_embeds = torch.cat(image_embeds,dim=0)

    print(text_embeds.shape, image_embeds.shape)
    sims_matrix = text_embeds @ image_embeds.t()
    # return sims_matrix.cpu().numpy()

    # score_matrix_t2i = torch.full((len(texts), len(images)),-100.0).to(device)
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0),start+step)
    print(sims_matrix[start:end].shape)
    for i,sims in tqdm(enumerate(sims_matrix[start:end])): 
        topk_sim, topk_idx = sims.topk(topk, dim=0)
        # print(f'image feats device: {image_feats.device}, topk_sim device: {topk_sim.device}, topk idx device:{topk_idx.device}')
        encoder_output = image_feats[topk_idx.cpu()]
        encoder_output = encoder_output.to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[start+i].repeat(topk,1), 
                                    attention_mask = text_atts[start+i].repeat(topk,1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,
                                    return_dict = True)
        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        sims_matrix[start+i,topk_idx] = score + topk_sim

    return sims_matrix.cpu().numpy()
    
def eval(similarity, idx_dic,score_list):
    similarity_argsort = np.argsort(-similarity, axis=1)
    print('similarity_argsort.shape', similarity_argsort.shape)

    def func_check_same_attri(s1, s2):
        s1 = s1.split(',')
        s2 = s2.split(',')
        if len(s1) != 21: return 0
        if s1[0] != s2[0]:
            return 0
        same = 0
        sum1 = 0
        for i in range(len(s1)):
            if s1[i] == '1' and s2[i] == '1':
                same += 1
            if s1[i] == '1': sum1 += 1
        # print(same, sum1, same==sum1)
        return same == sum1

    tp = 0
    map_ori = 0
    map_debug = 0

    topk = 10
    for i in range (len(similarity_argsort)):
        # print(i, similarity_argsort[i][:topk])
        get = 1e-6
        ap = 0
        for j in range(topk):
            # print(i, similarity_argsort[i][j], idx_dic[i], idx_dic[similarity_argsort[i][j]], func_check_same_attri(idx_dic[i], idx_dic[similarity_argsort[i][j]]))
            if (idx_dic[i] == idx_dic[similarity_argsort[i][j]]) or func_check_same_attri(idx_dic[i], idx_dic[similarity_argsort[i][j]]):
                get += 1
                # print(get, j + 1, similarity_argsort[i][j])
                ap += (get / (j+1))

        ap_ori = ap/get
        ap_debug = ap/topk

        map_ori = map_ori + ap_ori
        map_debug = map_debug + ap_debug

    score_list.append([map_ori/len(similarity_argsort), map_debug/len(similarity_argsort)])
    print('score ori:', map_ori/len(similarity_argsort))
    print('score debug:', map_debug/len(similarity_argsort))
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained_list = "/home/wangjing/wanJ_workspace/BLIP/output/vit_base_prompt_lr_transforms1"
    image_size = 224
    images_root = '/home/wangjing/wanJ_workspace/CVPR2023_foundation_model_Track2/OneForAll/datasets/val/val_images/'
    texts_root = '/home/wangjing/wanJ_workspace/CVPR2023_foundation_model_Track2/OneForAll/datasets/val/val_label.txt'
    score_list = []
    for pretrained_name in sorted(os.listdir(pretrained_list)[:]):
        if pretrained_name.endswith('.pth'):
            print(pretrained_name)

            model = blip_retrieval(pretrained=os.path.join(pretrained_list, pretrained_name), image_size=image_size, vit='base', 
                                    vit_grad_ckpt=True, vit_ckpt_layer=4, 
                                    queue_size=57600, negative_all_rank=True)
            # model = blip_retrieval(pretrained=os.path.join(pretrained_list, pretrained_name), image_size=image_size, vit='large', 
            #                        vit_grad_ckpt=True, vit_ckpt_layer=12, queue_size=57600, 
            #                        negative_all_rank=True)
            model = model.to(device)

            transform_ops = get_transforms()
            images, texts, idx_dic = get_images_texts(images_root, texts_root, transform_ops)
            similarity = get_similarity(model, images, texts, device, topk=10)
            eval(similarity, idx_dic, score_list)
    print(score_list)
    




