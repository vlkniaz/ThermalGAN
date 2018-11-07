import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
import cv2
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from data.image_folder import make_thermal_dataset
from PIL import Image


class ThermalRelDataset(BaseDataset):
    def initialize(self, opt):
        print('ThermalRelDataset')
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = make_thermal_dataset(self.dir_AB)
        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        A_path = self.AB_paths[index]['A']
        B_path = self.AB_paths[index]['B']
        LABEL_path = self.AB_paths[index]['LABEL']
        
        A = Image.open(A_path).convert('RGB')
        A = A.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        A = transforms.ToTensor()(A)
        
        B = np.load(B_path)
        B = cv2.resize(B, dsize=(self.opt.loadSize, self.opt.loadSize), interpolation=cv2.INTER_CUBIC) 

        # Create relative temperature B
        LABEL = np.load(LABEL_path)
        LABEL = cv2.resize(LABEL, dsize=(self.opt.loadSize, self.opt.loadSize), interpolation=cv2.INTER_CUBIC) 
        #print('Before')
        #print(B)
        B = B - LABEL
        #print('After')
        #print(B)

        B = np.reshape(B, (self.opt.loadSize, self.opt.loadSize, 1))
        B = np.clip(B, -10.0, 10.0)
        B = B / 10.0
        B = torch.from_numpy(B.transpose((2, 0, 1))).float()
        
        # Create normalized thermal segmentation
        LABEL = np.reshape(LABEL, (self.opt.loadSize, self.opt.loadSize, 1))
        #LABEL = np.clip(LABEL, -30.0, 30.0)
        #LABEL = LABEL / 30.0
        LABEL = np.clip(LABEL, -40.0, 40.0)
        LABEL = LABEL / 40.0
        LABEL = torch.from_numpy(LABEL.transpose((2, 0, 1))).float()

        w_total = A.size(2)
        w = int(w_total)
        h = A.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = A[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = B[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        LABEL = LABEL[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        # Create 4D input Thermal segmentation + RGN 
        #print(A)
        #print(LABEL)
        A = torch.cat((LABEL, A), 0)
        #print(A)
        #B = transforms.Normalize([0.0], [50.0])(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        #if (not self.opt.no_flip) and random.random() < 0.5:
        #    idx = [i for i in range(A.size(2) - 1, -1, -1)]
        #    idx = torch.LongTensor(idx)
        #    A = A.index_select(2, idx)
        #    B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        #if output_nc == 1:  # RGB to gray
        #    tmp = B
        #    B = tmp.unsqueeze(0)
        #print('A')
        #print(A)
        #print('B')
        #print(B)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'ThermalDataset'
