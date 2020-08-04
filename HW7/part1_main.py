'''
AIHW 7

Ivan Christian
'''

import os
import torch
import numpy as np

from hwhelpers.utils.imgnetdatastuff import *

from hwhelpers.utils.guidedbpcodehelpers import *

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



def get_percentile(model_name, conv0):
	'''
	Args:
	- model_name : string (nmae of the model for the key )
	'''
	conv0 = np.array(conv0)

	percentile_list = [i for i in range(5,100,5)]
	print(percentile_list)

	model_dict = defaultdict(lambda: defaultdict(list))

	for percent in percentile_list:

		model_dict[model_name][percent].extend([np.percentile(conv0, percent)])

	return model_dict



def get_weights(vgg_model, vgg_bn_model, data_loader, criterion, device):
	global inp

	def hook_fn(m, i, o):
		mod1 = m.in_channels==3 and m.out_channels==64  # conv module closest to input
		mod2 = m.in_channels==64 and m.out_channels==64 # conv module 2nd closest to input
		if mod1:
			end = "_1"
		elif mod2:
			end = "_2"
		if mod1 or mod2:
			global inp

			# print(inp['filename'][0].split('\\')[2][:-5])
			filename = inp['filename'][0].split('\\')[2][:-5]

			if not os.path.exists('vgg16'):
				os.makedirs('vgg16')

			for grad in o:
				try:
					l2_norm = grad[0].norm()  # output[0]
					torch.save(l2_norm, './vgg16/'+filename+end)
				except AttributeError: 
					print ("None found for Gradient")

	def hook_fn_bn(m, i, o):
		mod1 = m.in_channels==3 and m.out_channels==64  # conv module closest to input
		mod2 = m.in_channels==64 and m.out_channels==64 # conv module 2nd closest to input
		if mod1:
			end = "_1"
		elif mod2:
			end = "_2"
		if mod1 or mod2: 
			global inp

			filename = inp['filename'][0].split('\\')[2][:-5]

			if not os.path.exists('vgg16_bn'):
				os.makedirs('vgg16_bn')
			for grad in o:
				try:
					l2_norm = grad[0].norm()   # sum across all channels and dimensions
					torch.save(l2_norm, './vgg16_bn/'+filename+end)
					# print(l2_norm)
				except AttributeError: 
					print ("None found for Gradient")
	for module in vgg_model.modules():
		if isinstance(module,nn.Conv2d):   # only get the gradients of conv modules
			module.register_backward_hook(hook_fn)

	for module in vgg_bn_model.modules():
		if isinstance(module,nn.Conv2d):   # only get the gradients of conv modules
			module.register_backward_hook(hook_fn_bn)

	for idx, inp in enumerate(tqdm(data_loader)):
		img = inp['image'].to(device)
		label = inp['label'].to(device)
		filename = inp['filename'][0].split('\\')[2][:-5]

		out = vgg_model(img)
		out_bn = vgg_bn_model(img)
		
		loss = criterion(out, label)
		loss.backward()

		loss_bn = criterion(out_bn, label)
		loss_bn.backward()
	print('Finished collectimg the weights')

def plot():
	l2_ls = list()
	l2_bn_ls = list()

	for filename in os.listdir("vgg16"):
		l2_ls.append(torch.load(os.path.join("vgg16",filename)).item())

	for filename in os.listdir("vgg16_bn"):
		l2_bn_ls.append(torch.load(os.path.join("vgg16_bn",filename)).item())


	vgg_dict = get_percentile('vgg16', l2_ls)

	pprint(vgg_dict)

	vgg_bn_dict = get_percentile('vgg16_bn', l2_bn_ls)

	pprint(vgg_bn_dict)

	plt.plot(range(0,len(l2_ls)),sorted(l2_ls),label='vgg16')
	plt.plot(range(0,len(l2_bn_ls)),sorted(l2_bn_ls),label='vgg16_bn')
	plt.legend()
	plt.xlabel('N Samples')
	plt.ylabel('L2 Norm')
	plt.show()


def run():
	
	device = 'cuda' if torch.cuda.is_available else 'cpu'
	im_path = os.path.join('imgnet500', 'imagespart') # folder containing thee images
	synsetfile = os.path.join('hwhelpers', 'synset_words.txt')
	labels_path = os.path.join('hwhelpers', 'val')

	vgg_model = vgg16(pretrained = True, progress = True).to(device)
	vgg_bn_model = vgg16_bn(pretrained = True, progress = True).to(device)

	data_loader = create_loader(im_path, labels_path, synsetfile)

	criterion = nn.CrossEntropyLoss()

	get_weights(vgg_model, vgg_bn_model, data_loader, criterion, device)

	plot()

	



if __name__ =='__main__':
	run()

