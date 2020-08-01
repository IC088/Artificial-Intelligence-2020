import os
import torch
import numpy as np


from hwhelpers.utils.dataloader import *
from hwhelpers.utils.newrelu import *

from hwhelpers.imgnetdatastuff import *

from hwhelpers.guidedbpcodehelpers import *

from torchvision import datasets, models, transforms

from torchvision.models import vgg16, vgg16_bn

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn

import pickle
import json
from collections import defaultdict


from pprint import pprint

from tqdm import tqdm


def create_loader(im_path, labels_path, synsetfile):
	'''
	Args:

	- im_path: string (file path containing the images)

	- synsetfile : string (file name for the synsetfile)

	returns:

	- data_loader : DataLaoder 
	'''

	transform = transforms.Compose([transforms.Resize((224, 224)), 
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

	dataset = dataset_imagenetvalpart(im_path, labels_path, synsetfile, 250 , transform=transform)
	# dataset = ImageNetDataset(im_path, synsetfile, transform=transform)

	data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

	return data_loader



def get_guided_bp(model, dataloader, device):
	inp = next(iter(dataloader))
	img = inp['image'].to(device)
	img.requires_grad=True

	out = model(img)

	criterion = nn.CrossEntropyLoss()
	loss = criterion(out, inp['label'].to(device))
	loss.backward()
	imshow2(img.grad,img.to(device))


def run():
	device = 'cuda' if torch.cuda.is_available else 'cpu'
	im_path = os.path.join('imgnet500', 'imagespart') # folder containing thee images
	labels_path = os.path.join('hwhelpers', 'val')
	synsetfile = os.path.join('hwhelpers', 'synset_words.txt')

	vgg16_model = vgg16(pretrained = True, progress = True).to(device)

	for name,module in vgg16_model.named_modules():
		if isinstance(module,nn.ReLU):  # if module is relu, change to our custom relu
			setbyname(vgg16_model,name,MyReLU())
	data_loader = create_loader(im_path, labels_path,synsetfile)

	get_guided_bp(vgg16_model, data_loader, device)


if __name__ == '__main__':
	run()