import json
import argparse
import os
import torch
import torchvision
from extract_conv_feats import Res
from extract_conv_feats import CNN
from extract_conv_feats import BlocksModule
#from wordnet_utils import myWordnet
import skimage
import skimage.io
import numpy as np
import random
import h5py


def create_vocabulary(images, word_min):
	counter = {}
	for image in images:
		for s in image['sentences']:
			for w in s['tokens']:
				counter[w] = counter.get(w, 0) + 1

	len_counter = {}
	for image in images:
		for s in image['sentences']:
			tokens = s['tokens']
			length = len(tokens)
			len_counter[length] = len_counter.get(length, 0) + 1

	low_freq_words = [w for w,n in counter.items() if n <= word_min]
	vocab = [w for w,n in counter.items() if n > word_min]
	low_freq_words_count = sum(counter[w] for w in low_freq_words)
	if low_freq_words_count > 0:
		vocab.append('UNK')

	for image in images:
		image['final_captions'] = []
		for s in image['sentences']:
			tokens = s['tokens']
			caption = [w if counter.get(w,0) > word_min else 'UNK' for w in tokens]
			image['final_captions'].append(caption)

	return vocab

def get_data(images, max_word, word2index):
	num_image = len(images)
	num_caption = sum(len(image['final_captions']) for image in images)
	
	sent_arrays = []
	sent_start_ix = np.zeros(num_image, dtype='uint32')
	sent_end_ix = np.zeros(num_image, dtype='uint32')
	sent_length = np.zeros(num_caption, dtype='uint32')
	c_c = 0
	c = 1
	for i,image in enumerate(images):
		n = len(image['final_captions'])
		length_i = np.zeros((n, max_word), dtype='uint32')
		for j,s in enumerate(image['final_captions']):
			sent_length[c_c] = min(max_word, len(s))
			c_c += 1
			for k,w in enumerate(s):
				if k < max_word:
					length_i[j,k] = word2index[w]

		sent_arrays.append(length_i)
		sent_start_ix[i] = c
		sent_end_ix[i] = c + n - 1
		
		c += n

	S = np.concatenate(sent_arrays, axis=0)
	return S, sent_start_ix, sent_end_ix, sent_length

def process_json(json_path, args):
	with open(json_path) as f:
		images = json.load(f)['images']

	random.seed(789)

	vocabulary = create_vocabulary(images, args['word_min'])
	index2word = {i+1:w for i,w in enumerate(vocabulary)}
	word2index = {w:i+1 for i,w in enumerate(vocabulary)}

	S, sent_start_ix, sent_end_ix, sent_length = get_data(images, args['max_word'], word2index)

	num_image = len(images)
	ff = h5py.File(args['output_caption']+'.h5', "w")
	ff.create_dataset("sent", dtype='uint32', data=S)
	ff.create_dataset("sent_start_ix", dtype='uint32', data=sent_start_ix)
	ff.create_dataset("sent_end_ix", dtype='uint32', data=sent_end_ix)
	ff.create_dataset("sent_length", dtype='uint32', data=sent_length)
	ff.close()

	output = {}
	output['index_to_word'] = index2word
	output['images'] = []
	for i,image in enumerate(images):
		temp = {}
		temp['split'] = image['split']
		if 'filename' in image: temp['file_path'] = os.path.join(image['split'], image['filename'])
		if 'cocoid' in image: temp['id'] = image['cocoid']
		output['images'].append(temp)

	with open(args['output_json'], 'w') as f:
		json.dump(output, f)

def get_cnn_features(json_path, args):
	#wordnet = myWordnet()
	#wordnet.load_state_dict('./data/wordnet')
	cn = CNN(BlocksModule, [3, 4, 23, 3])
	cn.load_state_dict(torch.load(os.path.join('./data/res','res1.pth')))
	resnet = Res(cn)
	resnet.cuda()
	resnet.eval()

	with open(json_path) as f:
		images = json.load(f)['images']

	num_image = len(images)

	random.seed(789)

	global_feature_path = args['output_path'] + '_global'
	local_feature_path = args['output_path'] + '_local'
	#word_feature_path = args['output_path'] + '_word'

	if not os.path.isdir(global_feature_path):
		os.mkdir(global_feature_path)
	if not os.path.isdir(local_feature_path):
		os.mkdir(local_feature_path)
	#if not os.path.isdir(word_feature_path):
	#    os.mkdir(word_feature_path)

	for i,image in enumerate(images):

		temp = skimage.io.imread(os.path.join(args['image_path'], image['split'], image['filename']))
		if len(temp.shape) == 2:
			temp = temp[:,:,np.newaxis]
			temp = np.concatenate((temp,temp,temp), axis=2)

		temp = temp.astype('float32')/255.0
		temp = torch.from_numpy(temp.transpose([2,0,1])).cuda()
		temp = torch.autograd.Variable(torchvision.transforms.Compose([torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(temp), volatile=True)
		global_feature, local_feature = resnet(temp, 14)
		#word_feature = wordnet.forward(os.path.join(args['image_path'], image['split'], image['filename']))

		np.save(os.path.join(global_feature_path, str(image['cocoid'])), global_feature.data.cpu().float().numpy())
		#np.save(os.path.join(word_feature_path, str(image['cocoid'])), word_feature)
		np.savez_compressed(os.path.join(local_feature_path, str(image['cocoid'])), feat=local_feature.data.cpu().float().numpy())

		if i % 1 == 0:
		  print 'processing 1'


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Create the dataset bundles to train or test a model")

	parser.add_argument('--input', required=True, 
						help='input json')
	parser.add_argument("--word_min", type=int, default=5,
						help="word min")
	parser.add_argument("--max_word", type=int, default=16,
						help="max word")
	parser.add_argument('--output_caption', required=True, 
						help='output h5 caption')
	parser.add_argument('--output_json', required=True, 
						help='output json caption')
	parser.add_argument('--output_path', required=True, 
						help='output feature path')
	parser.add_argument('--image_path', required=True, 
						help='image path')

	arguments = parser.parse_args()

	arguments = vars(arguments)

	process_json(arguments['input'], arguments)

	get_cnn_features(arguments['input'], arguments)
