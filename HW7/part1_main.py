'''
AIHW 7

Ivan Christian
'''

import os
import torch
import numpy as np


from hwhelpers.utils.dataloader import *

from hwhelpers.imgnetdatastuff import *

from hwhelpers.guidedbpcodehelpers import *

from torchvision import datasets, models, transforms

from torchvision.models import vgg16, vgg16_bn

from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from torch.autograd import Variable

import pickle
import json
from collections import defaultdict


import torch.nn as nn

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

	data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

	return data_loader



def get_percentile(model_path,model_name, conv0, conv1):
	conv0 = np.array(conv0)
	conv1 = np.array(conv1)

	percentile_list = [i for i in range(5,100,5)]
	print(percentile_list)

	model_dict = defaultdict(lambda: defaultdict(list))

	for percent in percentile_list:

		model_dict[model_name][percent].extend([np.percentile(conv0, percent), np.percentile(conv1, percent)])
	return model_dict


class backward_hook():
	def __init__(self, module):
		self.hook = module.register_backward_hook(self.hook_fn)
	
	def hook_fn(self, module, grad_input, grad_output):
		self.norm = grad_output[0].norm()
	
	def close(self):
		self.hook.remove()

def run():
	device = 'cuda' if torch.cuda.is_available else 'cpu'
	im_path = os.path.join('imgnet500', 'imagespart') # folder containing thee images
	synsetfile = os.path.join('hwhelpers', 'synset_words.txt')
	labels_path = os.path.join('hwhelpers', 'val')

	vgg_model = vgg16(pretrained = True, progress = True).to(device)
	vgg_bn_model = vgg16_bn(pretrained = True, progress = True).to(device)


	vgg16_hooks = [backward_hook(vgg_model.features[0]), backward_hook(vgg_model.features[2])]

	vgg16_bn_hooks = [backward_hook(vgg_bn_model.features[0]), backward_hook(vgg_bn_model.features[2])]

	data_loader = create_loader(im_path, labels_path, synsetfile)

	results = 'part1'
	model_vgg_name = 'vgg16'
	model_vggbn_name = 'vgg16_bn'
	if not os.path.exists(results):
		os.makedirs(results)

	if not os.path.exists(os.path.join(results, model_vgg_name)):
		os.makedirs(os.path.join(results, model_vgg_name))

	if not os.path.exists(os.path.join(results, model_vggbn_name)):
		os.makedirs(os.path.join(results, model_vggbn_name))

	criterion = nn.CrossEntropyLoss()

	for idx, inp in enumerate(tqdm(data_loader)):
		img = inp['image'].to(device)
		label = inp['label'].to(device)
		filename = inp['filename'][0].split('\\')[2][:-5]

		out = vgg_model(img)
		out_bn = vgg_bn_model(img)
		
		loss = criterion(out, label)
		loss.backward()

		with open(os.path.join( results, model_vgg_name ,filename + '.pkl') , 'wb') as f:
			pickle.dump(vgg16_hooks, f)
		
		loss_bn = criterion(out_bn, label)
		loss_bn.backward()

		with open(os.path.join( results, model_vggbn_name ,filename + '.pkl') , 'wb') as f:
			pickle.dump(vgg16_bn_hooks, f)



if __name__ =='__main__':
	run()

