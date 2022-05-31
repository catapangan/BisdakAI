"""
BisdakAI Heart Murmur Recognition
"""
from torch import nn

import ai8x

class AI85BISDAKAI2DNet(nn.Module):
	def __init__(
				self,
				num_classes=9,
				num_channels=64,
				dimensions=(64, 1),
				fc_inputs=128,
				bias=False,
				**kwargs
	):
		super().__init__()
		self.drop = nn.Dropout(p=0.2)
		
		# Initial dimension
		dim = dimensions[1]
		
		kernel = 1
		padding = 0
		stride = 1
		self.conv1 = ai8x.FusedConv1dReLU(num_channels, 100, 1, stride=stride, padding=padding, bias=bias, **kwargs)
		dim = ((dim - kernel + 2 * padding) / stride) + 1
		
		kernel = 3
		padding = 0
		stride = 1
		self.conv2 = ai8x.FusedConv1dReLU(100, 96, 3, stride=stride, padding=padding, bias=bias, **kwargs)
		dim = ((dim - kernel + 2 * padding) / stride) + 1
		
		kernel = 3
		padding = 1
		stride = 1
		pool_stride = 1
		pool_size = 2
		self.conv3 = ai8x.FusedMaxPoolConv1dReLU(96, 64, 3, stride=stride, padding=padding, pool_stride=pool_stride, pool_size=pool_size, bias=bias, **kwargs)
		dim = ((dim - kernel + 2 * padding) / stride) + 1
		dim = ((dim - pool_size) / pool_stride) + 1
		
		kernel = 3
		padding = 0
		stride = 1
		self.conv4 = ai8x.FusedConv1dReLU(64, 48, 3, stride=stride, padding=padding, bias=bias, **kwargs)
		dim = ((dim - kernel + 2 * padding) / stride) + 1
		
		kernel = 3
		padding = 1
		stride = 1
		pool_stride = 1
		pool_size = 2
		self.conv5 = ai8x.FusedMaxPoolConv1dReLU(48, 64, 3, stride=stride, padding=padding, pool_stride=pool_stride, pool_size=pool_size, bias=bias, **kwargs)
		dim = ((dim - kernel + 2 * padding) / stride) + 1
		dim = ((dim - pool_size) / pool_stride) + 1
		
		kernel = 3
		padding = 0
		stride = 1
		self.conv6 = ai8x.FusedConv1dReLU(64, 96, 3, stride=stride, padding=padding, bias=bias, **kwargs)
		dim = ((dim - kernel + 2 * padding) / stride) + 1
		
		kernel = 3
		padding = 1
		stride = 1
		pool_stride = 1
		pool_size = 2
		self.conv7 = ai8x.FusedAvgPoolConv1dReLU(96, 100, 3, stride=stride, padding=padding, pool_stride=pool_stride, pool_size=pool_size, bias=bias, **kwargs)
		dim = ((dim - kernel + 2 * padding) / stride) + 1
		dim = ((dim - pool_size) / pool_stride) + 1
		
		kernel = 3
		padding = 1
		stride = 1
		pool_stride = 1
		pool_size = 2
		self.conv8 = ai8x.FusedMaxPoolConv1dReLU(100, fc_inputs, 6, stride=stride, padding=padding, pool_stride=pool_stride, pool_size=pool_size, bias=bias, **kwargs)
		dim = ((dim - kernel + 2 * padding) / stride) + 1
		dim = ((dim - pool_size) / pool_stride) + 1
		
		self.fc = ai8x.Linear(fc_inputs, num_classes, bias=bias, wide=True, **kwargs)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.drop(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = self.drop(x)
		x = self.conv5(x)
		x = self.conv6(x)
		x = self.drop(x)
		x = self.conv7(x)
		x = self.conv8(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


def ai85bisdakai2dnet(pretrained=False, **kwargs):
	assert not pretrained
	return AI85BISDAKAI2DNet(**kwargs)


models = [
	{
		'name': 'ai85bisdakai2dnet',
		'min_input': 1,
		'dim': 2,
	},
]
