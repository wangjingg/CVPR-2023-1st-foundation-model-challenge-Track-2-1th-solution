import os
import json

from torch.utils.data import Dataset
from PIL import Image
from data.utils import pre_caption, padding_resize

class cvpr23_finetune(Dataset):
    def __init__(self, image_size, transform, image_root, ann_path, max_words=100, prompt=''):
        self.annotation = json.load(open(ann_path['train'],'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        self.image_size = image_size
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root,ann['image'])
        image = Image.open(image_path).convert('RGB') 
        image = padding_resize(image, self.image_size)
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']] 
     
class paddle_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_path, split, max_words=100):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        self.annotation = json.load(open(ann_path[split],'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = padding_resize(image, 224)
        image = self.transform(image)  

        return image, index
    
class paddle_karpathy_train(Dataset):
    def __init__(self, transform, image_root, ann_path, max_words=100, prompt=''):        
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        '''        
        self.annotation = json.load(open(ann_path['train'],'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        
        self.img_ids = {}  
        n = 0
        for ann in self.annotation:
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1    
        
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = padding_resize(image, 224)

        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 

        return image, caption, self.img_ids[ann['image_id']] 
    
class paddle_karpathy_eval(Dataset):
    def __init__(self, transform, image_root, ann_path, split, max_words=100):  
        '''
        image_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        self.annotation = json.load(open(ann_path[split],'r'))
        self.transform = transform
        self.image_root = image_root
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])        
        image = Image.open(image_path).convert('RGB')    
        image = padding_resize(image, 224)
        image = self.transform(image)  

        return image, index

class cvpr23_pretrain(Dataset):
    def __init__(self, image_size, transform, data_root, image_list, label_list, max_words=100, prompt=''):        
        '''
        data_root (string): Root directory of images (e.g. coco/images/)
        ''' 
        
        self.annot_list = []
        for annot in label_list:
            self.annot_list.append(json.load(open(os.path.join(data_root, annot),'r')))
        self.transform = transform
        self.image_root = data_root
        self.max_words = max_words      
        self.prompt = prompt
        self.image_list = image_list
        self.image_size = image_size
        
        self.img_ids = {}  
        n = 0
        for part_idx, part_annot in enumerate(self.annot_list):
            for ann_idx, ann in enumerate(part_annot):
                self.annot_list[part_idx][ann_idx]['image_id'] = str(part_idx) + "_" + ann['image_id']
                self.annot_list[part_idx][ann_idx]['image'] = os.path.join(self.image_list[part_idx], ann['image'])
                img_id = self.annot_list[part_idx][ann_idx]['image_id']
                if img_id not in self.img_ids.keys():
                    self.img_ids[img_id] = n
                    n += 1

        self.annotation = []
        for annot in self.annot_list:
            self.annotation.extend(annot)

    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])
        image = Image.open(image_path).convert('RGB')   
        image = padding_resize(image, self.image_size)
        image = self.transform(image)
        
        caption = self.prompt+pre_caption(ann['caption'], self.max_words)
        return image, caption, self.img_ids[ann['image_id']]



