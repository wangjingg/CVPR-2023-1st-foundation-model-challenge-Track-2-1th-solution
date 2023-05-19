import os
import  torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from data.utils import padding_resize

class cmirt_dataset_image(Dataset):
    def __init__(self, image_dir, image_size, device):
        '''
        image_root (string): Root directory of images
        ann_root (string): directory to store the annotation file
        split (string): train, val or test
        '''
        # urls = {'train': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/nlvr_train.json',
        #         'val': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/nlvr_dev.json',
        #         'test': 'https://storage.googleapis.com/sfr-vision-language-research/datasets/nlvr_test.json'}
        filenames = {'train': 'train/train_label.txt', 'val': 'val/val_label.txt', 'test': 'test/test_text.txt'}

        self.image_dir = image_dir
        self.image_list = sorted(os.listdir(image_dir))
        self.image_size = image_size
        # self.device = device
        self.transform = transforms.Compose([
            # transforms.Resize((self.image_size, self.image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):

        image_name = self.image_list[index]
        image_path = os.path.join(self.image_dir, image_name)
        image = self._load_image(image_path)
        image = image
        return image_name, image,


    def _load_image(self, image_path):
        raw_image = Image.open(image_path).convert('RGB')
        raw_image = padding_resize(raw_image, 224, color=(0, 0, 0))
        # w, h = raw_image.size

        image = self.transform(raw_image)
        return image