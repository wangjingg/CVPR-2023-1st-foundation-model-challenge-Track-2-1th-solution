# !/usr/bin/env python3
import os
import numpy as np
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.blip_retrieval import blip_retrieval
import utils
from torchvision import transforms
from PIL import Image
from data.utils import padding_resize

def get_transforms():
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
        ])
    return transform_test

def get_images_texts(images_root, texts_root, transform_ops, test=False):
    images = []
    texts = []

    f = open(texts_root, 'r')
    for line in f.readlines():
        text = line.strip()
        texts.append(text)
    idx_dic = {}
    cnts = 0
    images_names = sorted(os.listdir(images_root))
    for image_name in images_names:
        img = Image.open(os.path.join(images_root, image_name)).convert('RGB')    
        img = padding_resize(img, 224, color=(0,0,0))
        img = transform_ops(img)
        images.append(img)
        idx_dic[cnts] = image_name
        cnts += 1

    if test:
        images = images[:20]
        texts = texts[:20]
    return images, texts, idx_dic

@torch.no_grad()
def get_similarity(model, images, texts, device, rank=10, clip_list = []):
    # test
    model.eval() 
    print('Computing features for evaluation...')
    num_text = len(texts)
    num_image = len(images)
    batch_size = 100
    text_embeds = []
    image_embeds = []
    image_feats = []
    text_ids = []
    text_atts = [] 
    for i in tqdm(range(0, num_text, batch_size)):

        text = texts[i: min(num_text, i+batch_size)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=100, return_tensors="pt").to(device) 
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')  
        text_embed = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]))
        text_embeds.append(text_embed)
        text_ids.append(text_input.input_ids)
        text_atts.append(text_input.attention_mask)

    for i in tqdm(range(0, num_image, batch_size)):
        image = images[i: min(num_image, i+batch_size)]
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
    sims_matrix = text_embeds @ image_embeds.t()
    sims_matrix = F.normalize(sims_matrix, dim=-1)
    
    # 合并clip
    for sim in clip_list:
        sims_matrix += F.normalize(torch.from_numpy(sim).to(device), dim=-1)
    sims_matrix = sims_matrix/(len(clip_list) + 1)
    
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0)//num_tasks + 1
    start = rank*step
    end = min(sims_matrix.size(0), start+step)
    for i,sims in tqdm(enumerate(sims_matrix[start:end])): 
        topk_sim, topk_idx = sims.topk(rank, dim=0)
        # print(f'image feats device: {image_feats.device}, topk_sim device: {topk_sim.device}, topk idx device:{topk_idx.device}')
        encoder_output = image_feats[topk_idx.cpu()]
        encoder_output = encoder_output.to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1],dtype=torch.long).to(device)
        output = model.text_encoder(text_ids[start+i].repeat(rank,1), 
                                    attention_mask = text_atts[start+i].repeat(rank,1),
                                    encoder_hidden_states = encoder_output,
                                    encoder_attention_mask = encoder_att,                             
                                    return_dict = True,
                                   )

        score = model.itm_head(output.last_hidden_state[:,0,:])[:,1]
        sims_matrix[start+i,topk_idx] = F.normalize(score, dim=-1) + topk_sim
    return sims_matrix.cpu().numpy()

def infer(similarity, idx_dic, texts, save_json):
    similarity_argsort = np.argsort(-similarity, axis=1)
    topk = 10
    result_list = []
    for i in tqdm(range(len(similarity_argsort))):
        dic = {'text': texts[i], 'image_names': []}
        for j in range(topk):
            dic['image_names'].append(idx_dic[similarity_argsort[i,j]])
        result_list.append(dic)
    with open(save_json, 'w') as f:
        f.write(json.dumps({'results': result_list}, indent=4))

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pretrained = "/home/aistudio/data/data218773/blip_checkpoint.pth"
    image_size = 224
    rank = 10
    images_root = '/home/aistudio/data/data218656/test/test_images/'
    texts_root = '/home/aistudio/data/data218656/test/test_text.txt'
    save_json = '/home/aistudio/work/BLIP/result_json/infer_json.json'
    clip_1 = np.load('/home/aistudio/work/open_clip/vit-h-14-clip1.npy')
    clip_2 = np.load('/home/aistudio/work/open_clip/vit-h-14-clip2.npy')
    clip_3 = np.load('/home/aistudio/work/open_clip/xml-robert-vit-h-14-clip3.npy')

    model = blip_retrieval(pretrained=pretrained, image_size=image_size, vit='large', 
                           vit_grad_ckpt=True, vit_ckpt_layer=12, queue_size=57600, 
                           negative_all_rank=True)

    model = model.to(device)

    transform_ops = get_transforms()
    images, texts, idx_dic = get_images_texts(images_root, texts_root, transform_ops)
    similarity = get_similarity(model, images, texts, device, rank=rank, clip_list=[clip_1, clip_2, clip_3])
    infer(similarity, idx_dic, texts, save_json)




    
    