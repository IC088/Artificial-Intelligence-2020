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


from pprint import pprint

from tqdm import tqdm



def create_loader(im_path, synsetfile):
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


	dataset = ImageNetDataset(im_path, synsetfile, transform=transform)

	data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)

	return data_loader





class backward_hook():
    def __init__(self, module):
        self.hook = module.register_backward_hook(self.hook_fn)
    
    def hook_fn(self, module, grad_input, grad_output):
        self.norm = grad_output[0].norm()
    
    def close(self):
    	self.hook.remove()


def get_hooks(model_path,model, data_loader, device):
	model.eval()

	

	model_hooks = [backward_hook(model.features[0]), backward_hook(model.features[2])]

	conv0 = []
	conv1 = []

	for idx, batch in tqdm(enumerate(data_loader)):

		if idx == 249:
			print('reached 250')
			break

		x = Variable(batch['image'], requires_grad = True).to(device)

		filename = batch['filename'][0][:-5]
		out = model(x)
		gradients = torch.ones_like(out)
		out.backward(gradients)

		conv0.append(model_hooks[0].norm)
		conv1.append(model_hooks[1].norm)

		with open(os.path.join(model_path,filename + 'pkl'),'wb') as f:
			pickle.dump(model_hooks, f)
	return conv0, conv1


def get_percentile(model_path,model_name, conv0, conv1):
	conv0 = np.array(conv0)
	conv1 = np.array(conv1)

	percentile_list = [i for i in range(5,100,5)]
	print(percentile_list)

	model_dict = defaultdict(lambda: defaultdict(list))

	for percent in percentile_list:

		model_dict[model_name][percent].extend([np.percentile(conv0, percent), np.percentile(conv1, percent)])
	return model_dict








def run():
	device = 'cuda' if torch.cuda.is_available else 'cpu'
	im_path = os.path.join('imgnet500', 'imagespart') # folder containing thee images
	synsetfile = os.path.join('hwhelpers', 'synset_words.txt')

	vgg_model = vgg16(pretrained = True, progress = True).to(device)
	vgg_bn_model = vgg16_bn(pretrained = True, progress = True).to(device)


	# model_dict = {'vgg16' : vgg_model}
	model_dict = {'vgg16_bn': vgg_bn_model}

	data_loader = create_loader(im_path, synsetfile)

	results = 'part1'
	if not os.path.exists(results):
		os.makedirs(results)

	for model_name, model in model_dict.items():


		if not os.path.exists(os.path.join(results, model_name)):
			os.makedirs(os.path.join(results,model_name))

		model_path = os.path.join(results,model_name)
		conv0, conv1 = get_hooks(model_path, model, data_loader, device)

		#structure of the dict would be model_dict[model_name][nth percentile] -> [conv0 values, conv1 values]
		model_dict = get_percentile(model_path, model_name, conv0, conv1)

		pprint(model_dict)
		print('Finished Part 1')
		torch.cuda.empty_cache()









if __name__ =='__main__':
	run()

