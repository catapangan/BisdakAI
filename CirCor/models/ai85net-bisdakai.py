"""
BisdakAI Heart Murmur Recognition
"""
from torch import nn

import ai8x

class AI85BISDAKAINet(nn.Module):
	def __init__(
				self,
				num_classes=9,
				num_channels=1,
				dimensions=(4000, 1),
				fc_inputs=60,
				bias=False,
				**kwargs
	):
		super().__init__()
		
		# Ensure square input
		assert dimensions[0] == dimensions[1]
		
		# Initial dimension
		dim = dimensions[0]
		
		kernel = 3
		padding = 1
		stride = 1
		pool_size = 2
		pool_stride = 2
		self.conv1 = ai8x.FusedMaxPoolConv1dReLU(num_channels, 64, kernel, pool_size=pool_size, pool_stride=pool_stride, padding=padding, stride=stride, bias=bias, **kwargs)
		dim = ((dim - kernel + 2 * padding) / stride) + 1
		dim = ((dim - pool_size) / pool_stride) + 1
		
		kernel = 3
		padding = 1
		stride = 1
		self.conv2 = ai8x.FusedConv1dReLU(64, 60, kernel, padding=padding, stride=stride, bias=bias, **kwargs)
		dim = ((dim - kernel + 2 * padding) / stride) + 1

		kernel = 3
		padding = 1
		stride = 1
		pool_size = 2
		pool_stride = 2
		self.conv3 = ai8x.FusedAvgPoolConv1dReLU(60, 60, kernel, pool_size=pool_size, pool_stride=pool_stride, padding=padding, stride=stride, bias=bias, **kwargs)
		dim = ((dim - kernel + 2 * padding) / stride) + 1
		dim = ((dim - pool_size) / pool_stride) + 1
		
		kernel = 3
		padding = 1
		stride = 1
		self.conv4 = ai8x.FusedConv1dReLU(60, 60, kernel, padding=padding, stride=stride, bias=bias, **kwargs)
		dim = ((dim - kernel + 2 * padding) / stride) + 1

		kernel = 3
		padding = 1
		stride = 1
		pool_size = 2
		pool_stride = 2
		self.conv5 = ai8x.FusedAvgPoolConv1dReLU(60, 60, kernel, pool_size=pool_size, pool_stride=pool_stride, padding=padding, stride=stride, bias=bias, **kwargs)
		dim = ((dim - kernel + 2 * padding) / stride) + 1
		dim = ((dim - pool_size) / pool_stride) + 1
		
		kernel = 3
		padding = 1
		stride = 1
		self.conv6 = ai8x.FusedConv1dReLU(60, fc_inputs, kernel, padding=padding, stride=stride, bias=bias, **kwargs)
		dim = ((dim - kernel + 2 * padding) / stride) + 1

		self.fc = ai8x.Linear(fc_inputs * dim * dim, num_classes, bias=True, **kwargs)

	def forward(self, x):  # pylint: disable=arguments-differ
		x = self.conv1(x)
		x = self.conv2(x)
		x = self.conv3(x)
		x = self.conv4(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


def ai85bisdakainet(pretrained=False, **kwargs):
	assert not pretrained
	return AI85BISDAKAINet(**kwargs)


models = [
	{
		'name': 'ai85bisdakainet',
		'min_input': 1,
		'dim': 1,
	},
]