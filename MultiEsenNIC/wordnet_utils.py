import cPickle as pickle
import tensorflow as tf
import numpy as np
from wordnet import Detector
import os
import skimage.io
import skimage.transform


class myWordnet():
	def __init__(self):
		pass
		
	def load_state_dict(self, model_root):
		self.images_tf = tf.placeholder( tf.float32, [None, 224, 224, 3], name="images")
		with open(os.path.join(model_root, 'words.pkl'), 'rb') as f:
			n_labels = len(pickle.load(f))
		detector = Detector(os.path.join(model_root, 'caffe_layers_value.pickle'), n_labels)
		c1, c2, c3, c4, conv5, conv6, gap, output = detector.inference(self.images_tf)
		self.sig_output = tf.nn.sigmoid(output)
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		self.sess = tf.InteractiveSession(config=config)
		saver = tf.train.Saver()
		saver.restore(self.sess, os.path.join(model_root, 'wordnet'))

	def _load_image(self, img_path):
		try:
			img = skimage.io.imread( img_path ).astype( float )
		except:
			return None

		if img is None: return None
		if len(img.shape) < 2: return None
		if len(img.shape) == 4: return None
		if len(img.shape) == 2: img=np.tile(img[:,:,None], 3)
		if img.shape[2] == 4: img=img[:,:,:3]
		if img.shape[2] > 4: return None

		img /= 255.

		resized_img = skimage.transform.resize( img, [224,224] )
		return resized_img

	def forward(self, img_path):
		current_images = np.expand_dims(np.array(self._load_image(img_path)), axis=0)
		sig_output_val = self.sess.run(
				[self.sig_output],
				feed_dict={
					self.images_tf: current_images
					})

		return sig_output_val[0][0]
