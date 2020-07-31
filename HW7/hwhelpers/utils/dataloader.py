'''
Dataet loader
'''
import os
from torch.utils.data import Dataset

import os
from PIL import Image
import numpy as np
import pandas as pd


'''
Implement Dataset class to load images
'''

class ImageNetDataset(Dataset):

	def __init__(self, im_dir, synsetfile, transform = None):
		# self.root_dir = root

		self.im_dir = im_dir

		self.imgfilenames= self._get_filenames(self.im_dir)

		indicestosynsets,synsetstoindices,synsetstoclassdescriptions = self._parsesynsetwords(synsetfile)

		self.transform = transform

		

	def _get_filenames(self,path):

		all_files = []

		for root, dirs, file in os.walk(path):
			all_files.extend(file)
		all_files = sorted(all_files)

		return all_files



		

	def _parsesynsetwords(self,filen):

		synsetstoclassdescriptions={}
		indicestosynsets={}
		synsetstoindices={}
		ct=-1
		with open(filen) as f:
			for line in f:
				if (len(line)> 5):
					z=line.strip().split()
					descr=''
					for i in range(1,len(z)):
						descr=descr+' '+z[i]
			ct+=1
			indicestosynsets[ct]=z[0]
			synsetstoindices[z[0]]=ct
			synsetstoclassdescriptions[z[0]]=descr[1:]
		return indicestosynsets,synsetstoindices,synsetstoclassdescriptions



	def _load_image(self, image_name):
		'''
		Load image after getting the image from the files. Used for this class operation
		Inputs: the image name 
		Outputs: Image from the array
		'''
		image = Image.open(image_name)
		image.load()
		image = np.array(image)


		if len(image.shape) == 2:
			image = np.expand_dims(image, 2)
			image = np.repeat(image, 3, 2)

		return Image.fromarray(image)
	def __len__(self):
		return len(self.imgfilenames)

	def __getitem__(self, idx):
		image = self._load_image(os.path.join(self.im_dir, self.imgfilenames[idx]))
		# image = PIL.Image.open(self.imgfilenames[idx]).convert('RGB')
		# label = self.labels[idx]
		if self.transform:
			image = self.transform(image)
		sample = {'image': image, 'filename': self.imgfilenames[idx]}
		return sample