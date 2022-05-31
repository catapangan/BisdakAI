###################################################################################################
#
# Copyright (C) 2019-2022 Maxim Integrated Products, Inc. All Rights Reserved.
#
# Maxim Integrated Products, Inc. Default Copyright Notice:
# https://www.maximintegrated.com/en/aboutus/legal/copyrights.html
#
###################################################################################################
#
# Portions Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import errno
import hashlib
import os
import tarfile
import time
import urllib
import warnings
import shutil

import numpy as np
import torch
from torch.utils.model_zoo import tqdm
from torchvision import transforms

import librosa
import pytsmod as tsm

import ai8x


class CirCor2D:
	fs = 4000

	class_dict = {'AV_OK': 0, 'AV_MR': 1, 'MV_OK': 2, 'MV_MR': 3, 'PV_OK': 4, 'PV_MR': 5,
				  'TV_OK': 6, 'TV_MR': 7}

	def __init__(self, root, classes, d_type, t_type, transform=None, quantization_scheme=None,
				 augmentation=None, save_unquantized=False):

		self.root = root
		self.classes = classes
		self.d_type = d_type
		self.t_type = t_type
		self.transform = transform
		self.save_unquantized = save_unquantized

		self.__parse_quantization(quantization_scheme)
		self.__parse_augmentation(augmentation)

		if not self.save_unquantized:
			self.data_file = 'dataset2.pt'
		else:
			self.data_file = 'unquantized.pt'
			
		shutil.rmtree(os.path.join(self.root, self.__class__.__name__, 'processed'))
		if not os.path.exists(os.path.join(self.root, self.__class__.__name__, 'processed')):
			os.makedirs(os.path.join(self.root, self.__class__.__name__, 'processed'))
			
		self.__gen_datasets()

		self.data, self.targets, self.data_type = torch.load(os.path.join(
			self.processed_folder, self.data_file))

		print(f'\nProcessing {self.d_type}...')
		self.__filter_dtype()
		self.__filter_classes()

	@property
	def raw_folder(self):
		"""Folder for the raw data.
		"""
		return os.path.join(self.root, self.__class__.__name__, 'raw')

	@property
	def processed_folder(self):
		"""Folder for the processed data.
		"""
		return os.path.join(self.root, self.__class__.__name__, 'processed')

	def __parse_quantization(self, quantization_scheme):
		if quantization_scheme:
			self.quantization = quantization_scheme
			if 'bits' not in self.quantization:
				self.quantization['bits'] = 8
			if self.quantization['bits'] == 0:
				self.save_unquantized = True
			if 'compand' not in self.quantization:
				self.quantization['compand'] = False
			elif 'mu' not in self.quantization:
				self.quantization['mu'] = 255
		else:
			print('Undefined quantization schema! ',
				  'Number of bits set to 8.')
			self.quantization = {'bits': 8, 'compand': False}

	def __parse_augmentation(self, augmentation):
		self.augmentation = augmentation
		if augmentation:
			if 'aug_num' not in augmentation:
				print('No key `aug_num` in input augmentation dictionary! ',
					  'Using 0.')
				self.augmentation['aug_num'] = 0
			elif self.augmentation['aug_num'] != 0:
				if 'noise_var' not in augmentation:
					print('No key `noise_var` in input augmentation dictionary! ',
						  'Using defaults: [Min: 0., Max: 1.]')
					self.augmentation['noise_var'] = {'min': 0., 'max': 1.}
				if 'shift' not in augmentation:
					print('No key `shift` in input augmentation dictionary! '
						  'Using defaults: [Min:-0.1, Max: 0.1]')
					self.augmentation['shift'] = {'min': -0.1, 'max': 0.1}
				if 'strech' not in augmentation:
					print('No key `strech` in input augmentation dictionary! '
						  'Using defaults: [Min: 0.8, Max: 1.3]')
					self.augmentation['strech'] = {'min': 0.8, 'max': 1.3}

	def __gen_bar_updater(self):  # pylint: disable=no-self-use
		pbar = tqdm(total=None)

		def bar_update(count, block_size, total_size):
			if pbar.total is None and total_size:
				pbar.total = total_size
			progress_bytes = count * block_size
			pbar.update(progress_bytes - pbar.n)

		return bar_update

	def __filter_dtype(self):
		if self.d_type == 'train':
			idx_to_select = (self.data_type == 0)[:, -1]
		elif self.d_type == 'test':
			idx_to_select = (self.data_type == 1)[:, -1]
		else:
			print(f'Unknown data type: {self.d_type}')
			return

		self.data = self.data[idx_to_select, :]
		self.targets = self.targets[idx_to_select, :]
		del self.data_type

	def __filter_classes(self):
		initial_new_class_label = len(self.class_dict)
		new_class_label = initial_new_class_label
		for c in self.classes:
			if c not in self.class_dict:
				print(f'Class {c} not found in data')
				return
			num_elems = (self.targets == self.class_dict[c]).cpu().sum()
			print(f'Class {c} (# {self.class_dict[c]}): {num_elems} elements')
			self.targets[(self.targets == self.class_dict[c])] = new_class_label
			new_class_label += 1

		num_elems = (self.targets < initial_new_class_label).cpu().sum()
		print(f'Class UNKNOWN: {num_elems} elements')
		self.targets[(self.targets < initial_new_class_label)] = new_class_label
		self.targets -= initial_new_class_label

	def __len__(self):
		return len(self.data)

	def __getitem__(self, index):
		inp, target = self.data[index].type(torch.FloatTensor), int(self.targets[index])
		if not self.save_unquantized:
			inp /= 256
		if self.transform is not None:
			inp = self.transform(inp)
		return inp, target

	@staticmethod
	def add_white_noise(audio, noise_var_coeff):
		"""Adds zero mean Gaussian noise to image with specified variance.
		"""
		coeff = noise_var_coeff * np.mean(np.abs(audio))
		noisy_audio = audio + coeff * np.random.randn(len(audio))
		return noisy_audio

	@staticmethod
	def shift(audio, shift_sec, fs):
		"""Shifts audio.
		"""
		shift_count = int(shift_sec * fs)
		return np.roll(audio, shift_count)

	@staticmethod
	def stretch(audio, rate=1):
		"""Stretches audio with specified ratio.
		"""
		input_length = 4000
		audio2 = librosa.effects.time_stretch(audio, rate)
		if len(audio2) > input_length:
			audio2 = audio2[:input_length]
		else:
			audio2 = np.pad(audio2, (0, max(0, input_length - len(audio2))), "constant")

		return audio2

	def augment(self, audio, fs, verbose=False):
		"""Augments audio by adding random noise, shift and stretch ratio.
		"""
		random_noise_var_coeff = np.random.uniform(self.augmentation['noise_var']['min'],
												   self.augmentation['noise_var']['max'])
		random_shift_time = np.random.uniform(self.augmentation['shift']['min'],
											  self.augmentation['shift']['max'])
		random_strech_coeff = np.random.uniform(self.augmentation['strech']['min'],
												self.augmentation['strech']['max'])

		aug_audio = tsm.wsola(audio, random_strech_coeff)
		aug_audio = self.shift(aug_audio, random_shift_time, fs)
		aug_audio = self.add_white_noise(aug_audio, random_noise_var_coeff)
		if verbose:
			print(f'random_noise_var_coeff: {random_noise_var_coeff:.2f}\nrandom_shift_time: \
					{random_shift_time:.2f}\nrandom_strech_coeff: {random_strech_coeff:.2f}')
		return aug_audio

	def augment_multiple(self, audio, fs, n_augment, verbose=False):
		"""Calls `augment` function for n_augment times for given audio data.
		Finally the original audio is added to have (n_augment+1) audio data.
		"""
		aug_audio = [self.augment(audio, fs, verbose=verbose) for i in range(n_augment)]
		aug_audio.insert(0, audio)
		return aug_audio

	@staticmethod
	def compand(data, mu=255):
		"""Compand the signal level to warp from Laplacian distribution to uniform distribution"""
		data = np.sign(data) * np.log(1 + mu*np.abs(data)) / np.log(1 + mu)
		return data

	@staticmethod
	def expand(data, mu=255):
		"""Undo the companding"""
		data = np.sign(data) * (1 / mu) * (np.power((1 + mu), np.abs(data)) - 1)
		return data

	@staticmethod
	def quantize_audio(data, num_bits=8, compand=False, mu=255):
		"""Quantize audio
		"""
		if compand:
			data = CirCor2D.compand(data, mu)

		step_size = 2.0 / 2**(num_bits)
		max_val = 2**(num_bits) - 1
		q_data = np.round((data - (-1.0)) / step_size)
		q_data = np.clip(q_data, 0, max_val)

		if compand:
			data_ex = (q_data - 2**(num_bits - 1)) / 2**(num_bits - 1)
			data_ex = CirCor2D.expand(data_ex)
			q_data = np.round((data_ex - (-1.0)) / step_size)
			q_data = np.clip(q_data, 0, max_val)
		return np.uint8(q_data)

	def __gen_datasets(self, exp_len=4096, row_len=64, overlap_ratio=0):
		print('Generating dataset from raw data samples for the first time. ')
		print('This process will take significant time (~60 minutes)...')
		with warnings.catch_warnings():
			warnings.simplefilter('error')

			lst = sorted(os.listdir(self.raw_folder))
			labels = [d for d in lst if os.path.isdir(os.path.join(self.raw_folder, d))
					  and d[0].isalpha()]

			# PARAMETERS
			overlap = int(np.ceil(row_len * overlap_ratio))
			num_rows = int(np.ceil(exp_len / (row_len - overlap)))
			data_len = int((num_rows*row_len - (num_rows-1)*overlap))
			print(f'data_len: {data_len}')

			# show the size of dataset for each keyword
			print('------------- Label Size ---------------')
			for i, label in enumerate(labels):
				record_list = os.listdir(os.path.join(self.raw_folder, label))
				print(f'{label:8s}:  \t{len(record_list)}')
			print('------------------------------------------')

			for i, label in enumerate(labels):
				print(f'Processing the label: {label}. {i + 1} of {len(labels)}')
				record_list = sorted(os.listdir(os.path.join(self.raw_folder, label)))

				# dimension: row_length x number_of_rows
				if not self.save_unquantized:
					data_in = np.empty(((self.augmentation['aug_num'] + 1) * len(record_list),
										row_len, num_rows), dtype=np.uint8)
				else:
					data_in = np.empty(((self.augmentation['aug_num'] + 1) * len(record_list),
										row_len, num_rows), dtype=np.float32)
				data_type = np.empty(((self.augmentation['aug_num'] + 1) * len(record_list), 1),
									 dtype=np.uint8)
				# create data classes
				data_class = np.full(((self.augmentation['aug_num'] + 1) * len(record_list), 1), i,
									 dtype=np.uint8)

				time_s = time.time()
				train_count = 0
				test_count = 0
				for r, record_name in enumerate(record_list):
					if r % 1000 == 0:
						print(f'\t{r + 1} of {len(record_list)}')

					if hash(record_name) % 10 < 9:
						d_typ = np.uint8(0)  # train+val
						train_count += 1
					else:
						d_typ = np.uint8(1)  # test
						test_count += 1

					record_pth = os.path.join(self.raw_folder, label, record_name)
					record, fs = librosa.load(record_pth, offset=0, sr=None)
					audio_seq_list = self.augment_multiple(record, fs,
														   self.augmentation['aug_num'])
					for n_a, audio_seq in enumerate(audio_seq_list):
						# store set type: train+validate or test
						data_type[(self.augmentation['aug_num'] + 1) * r + n_a, 0] = d_typ

						# Write audio 128x128=16384 samples without overlap
						for n_r in range(num_rows):
							start_idx = n_r*(row_len - overlap)
							end_idx = start_idx + row_len
							audio_chunk = audio_seq[start_idx:end_idx]
							# pad zero if the length of the chunk is smaller than row_len
							audio_chunk = np.pad(audio_chunk, [0, row_len-audio_chunk.size])
							# store input data after quantization
							data_idx = (self.augmentation['aug_num'] + 1) * r + n_a
							if not self.save_unquantized:
								data_in[data_idx, :, n_r] = \
									CirCor2D.quantize_audio(audio_chunk,
													   num_bits=self.quantization['bits'],
													   compand=self.quantization['compand'],
													   mu=self.quantization['mu'])
							else:
								data_in[data_idx, :, n_r] = audio_chunk

				dur = time.time() - time_s
				print(f'Finished in {dur:.3f} seconds.')
				print(data_in.shape)
				time_s = time.time()
				if i == 0:
					data_in_all = data_in.copy()
					data_class_all = data_class.copy()
					data_type_all = data_type.copy()
				else:
					data_in_all = np.concatenate((data_in_all, data_in), axis=0)
					data_class_all = np.concatenate((data_class_all, data_class), axis=0)
					data_type_all = np.concatenate((data_type_all, data_type), axis=0)
				dur = time.time() - time_s
				print(f'Data concatenation finished in {dur:.3f} seconds.')

			data_in_all = torch.from_numpy(data_in_all)
			data_class_all = torch.from_numpy(data_class_all)
			data_type_all = torch.from_numpy(data_type_all)

			mfcc_dataset = (data_in_all, data_class_all, data_type_all)
			torch.save(mfcc_dataset, os.path.join(self.processed_folder, self.data_file))

		print('Dataset created.')
		print(f'Training+Validation: {train_count},  Test: {test_count}')

def CirCor2D_get_datasets(data, load_train=True, load_test=True, num_classes=8):
	"""
	Load the 1D samples of the CirCor Dataset

	The dataset is split into training+validation and test sets. 90:10 training+validation:test
	split is used by default.

	Data is augmented to 3x duplicate data by random stretch/shift and randomly adding noise where
	the stretching coefficient, shift amount and noise variance are randomly selected between
	0.8 and 1.3, -0.1 and 0.1, 0 and 1, respectively.
	"""
	(data_dir, args) = data

	transform = transforms.Compose([
		ai8x.normalize(args=args)
	])

	classes = next((e for _, e in enumerate(datasets)
						if len(e['output']) - 1 == num_classes))['output'][:-1]
	
	augmentation = {'aug_num': 2}
	quantization_scheme = {'compand': False, 'mu': 10}

	if load_train:
		train_dataset = CirCor2D(root=data_dir, classes=classes, d_type='train',
							transform=transform, t_type='keyword',
							quantization_scheme=quantization_scheme,
							augmentation=augmentation)
	else:
		train_dataset = None

	if load_test:
		test_dataset = CirCor2D(root=data_dir, classes=classes, d_type='test',
							transform=transform, t_type='keyword',
							quantization_scheme=quantization_scheme,
							augmentation=augmentation)

		if args.truncate_testset:
			test_dataset.data = test_dataset.data[:1]
	else:
		test_dataset = None

	return train_dataset, test_dataset

datasets = [
	{
		'name': 'CirCor2D',
		'input': (64, 64),
		'output': ('AV_OK', 'AV_MR', 'MV_OK', 'MV_MR', 'PV_OK', 'PV_MR', 'TV_OK', 'TV_MR', 'UNK'),
		'weight': (1, 1, 1, 1, 1, 1, 1, 1, 0.06),
		'loader': CirCor2D_get_datasets,
	},
]
