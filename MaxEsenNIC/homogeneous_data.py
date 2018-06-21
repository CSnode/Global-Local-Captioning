import numpy as np
import random
import atexit
import torch
import torch.utils.data
import time
import os
import json
import h5py


def np_load(arg1, arg2, arg3):
	return (np.load(arg2), np.load(arg3)['feat'], arg1)

class RawGET():
	def __init__(self, arg1, arg2, arg3=False):
		self.appq = arg1
		self.homogeneous = arg2
		self.rand = arg3

	def reset(self):
		self.appq_get = iter(torch.utils.data.DataLoader(dataset=self.homogeneous, batch_size=1, sampler=self.homogeneous.ixa_dead[self.appq][self.homogeneous.gga[self.appq]:], shuffle=False, pin_memory=True, num_workers=2, collate_fn=lambda faw: faw[0]))

	def _load_next_batch(self):
		max_app = len(self.homogeneous.ixa_dead[self.appq])
		win_flag = False

		rainif = self.homogeneous.gga[self.appq]
		eixa = self.homogeneous.ixa_dead[self.appq][rainif]

		rain_next = rainif + 1
		if rain_next >= max_app:
			rain_next = 0
			if self.rand:
				random.shuffle(self.homogeneous.ixa_dead[self.appq])
			win_flag = True
		self.homogeneous.gga[self.appq] = rain_next

		return eixa, win_flag
	
	def get(self):
		if hasattr(self, 'appq_get'):
			pass
		else:
			self.reset()

		ixa, win_flag = self._load_next_batch()
		md = self.appq_get.next()
		if win_flag:
			self.reset()

		return md + [win_flag]


class HomogeneousData(torch.utils.data.Dataset):

	def init_it(self, arg1):
		del self._app_prop[arg1]
		self._app_prop[arg1] = RawGET(arg1, self, arg1=='train')
		self.gga[arg1] = 0

	def banba_num(self):
		return self.banba_n

	def banba(self):
		return self.index_to_word

	def fetch_semini_num(self):
		return self.semini_num

	def __init__(self, arguments):
		self.arguments = arguments
		self.quo_num = self.arguments.mini_batch

		self.add = json.load(open(self.arguments.input_json))
		self.index_to_word = self.add['index_to_word']
		self.banba_num = len(self.index_to_word)
		
		self.input_captions = h5py.File(self.arguments.input_captions, 'r', driver='core')

		self.input_global_feature = self.arguments.input_global_feature
		self.input_local_feature = self.arguments.input_local_feature

		semini_num = self.input_captions['sent'].shape
		self.semini_num = semini_num[1]

		self.sent_start_ix = self.input_captions['sent_start_ix'][:]
		self.sent_end_ix = self.input_captions['sent_end_ix'][:]

		self.image_batch = self.sent_start_ix.shape[0]

		self.ixa_dead = {'train': [], 'val': [], 'test': []}
		for index in range(len(self.add['images'])):
			image = self.add['images'][index]
			if image['split'] == 'train':
				self.ixa_dead['train'].append(index)
			elif image['split'] == 'val':
				self.ixa_dead['val'].append(index)
			elif image['split'] == 'test':
				self.ixa_dead['test'].append(index)

		print('data: (%d, %d, %d)' % (len(self.ixa_dead['train']), len(self.ixa_dead['val']), len(self.ixa_dead['test'])))

		self.gga = {'train': 0, 'val': 0, 'test': 0}
		
		self._app_prop = {}
		for index in self.gga.keys():
			self._app_prop[index] = RawGET(index, self, index=='train')
			
		def terminal():
			print('Close RawGet')
			for index in self.ixa_dead.keys():
				del self._app_prop[index]
		atexit.register(terminal)

	def homogeneous_iter(self, arg1, arg2=None, arg3=None):
		homogeneous_num = arg2 or self.quo_num
		num_i = arg3 or 5

		global_feature_iter = []
		local_feature_iter = []
		input_caption_iter = np.zeros([homogeneous_num * num_i, self.semini_num + 2], dtype = 'int')
		musk_iter = np.zeros([homogeneous_num * num_i, self.semini_num + 2], dtype = 'float32')

		win_flag = False

		adds = []
		gta = []

		for i in range(homogeneous_num):

			wx_global_feature, wx_local_feature, wx_index, wx_win_flag = self._app_prop[arg1].get()
			global_feature_iter += [wx_global_feature] * num_i
			local_feature_iter += [wx_local_feature] * num_i

			index_start = self.sent_start_ix[wx_index] - 1 
			index_end = self.sent_end_ix[wx_index] - 1
			available_num = index_end - index_start + 1 

			if available_num < num_i:

				data_l = np.zeros([num_i, self.semini_num], dtype = 'int')
				for point in range(num_i):
					rand_index = random.randint(index_start,index_end)
					data_l[point, :] = self.input_captions['sent'][rand_index, :self.semini_num]
			else:
				rand_index = random.randint(index_start, index_end - num_i + 1)
				data_l = self.input_captions['sent'][rand_index: rand_index + num_i, :self.semini_num]
			
			input_caption_iter[i * num_i : (i + 1) * num_i, 1 : self.semini_num + 1] = data_l

			if wx_win_flag:
				win_flag = True

			gta.append(self.input_captions['sent'][self.sent_start_ix[wx_index] - 1: self.sent_end_ix[wx_index]])
		
			hash_table = {}
			hash_table['index'] = wx_index
			hash_table['id'] = self.add['images'][wx_index]['id']
			hash_table['file_path'] = self.add['images'][wx_index]['file_path']
			adds.append(hash_table)
		
		markble = np.array(list(map(lambda faw: (faw != 0).sum()+2, input_caption_iter)))
		for tmp_index, tmp_data in enumerate(musk_iter):
			tmp_data[:markble[tmp_index]] = 1

		result = {}
		result['global_feature'] = np.stack(global_feature_iter)
		result['local_feature'] = np.stack(local_feature_iter)
		result['captions'] = input_caption_iter
		result['gta'] = gta
		result['musk'] = musk_iter
		result['point'] = {'current_point': self.gga[arg1], 'max_point': len(self.ixa_dead[arg1]), 'end': win_flag}
		result['add'] = adds

		return result

	def __getitem__(self, arg1):

		index = arg1
		return np_load(index, \
				os.path.join(self.input_global_feature, str(self.add['images'][index]['id']) + '.npy'),
				os.path.join(self.input_local_feature, str(self.add['images'][index]['id']) + '.npz')
				)

	def __len__(self):
		return len(self.add['images'])
