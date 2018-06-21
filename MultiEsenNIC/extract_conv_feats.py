import torch
import torch.nn as nn
import math


def cv3(arg1, arg2, arg3=1):
	return nn.Conv2d(arg1, arg2, kernel_size=3, stride=arg3, padding=1, bias=False)

class Blocks(nn.Module):
	num = 1

	def __init__(self, arg1, arg2, arg3=1, arg4=None):
		super(Blocks, self).__init__()
		self.c1 = cv3(arg1, arg2, arg3)
		self.b1 = nn.BatchNorm2d(arg2)
		self.r = nn.ReLU(inplace=True)
		self.c2 = cv3(arg2, arg2)
		self.b2 = nn.BatchNorm2d(arg2)
		self.ds = arg4
		self.str = arg3

	def forward(self, x):
		block = x

		o = self.c1(x)
		o = self.b1(o)
		o = self.r(o)

		o = self.c2(o)
		o = self.b2(o)

		if self.ds is not None:
			block = self.ds(x)

		o += block
		o = self.r(o)

		return o

class BlocksModule(nn.Module):
	num = 4

	def __init__(self, arg1, arg2, arg3=1, arg4=None):
		super(BlocksModule, self).__init__()
		self.c1 = nn.Conv2d(arg1, arg2, kernel_size=1, stride=arg3, bias=False)
		self.b1 = nn.BatchNorm2d(arg2)
		self.c2 = nn.Conv2d(arg2, arg2, kernel_size=3, stride=1, padding=1, bias=False)
		self.b2 = nn.BatchNorm2d(arg2)
		self.c3 = nn.Conv2d(arg2, arg2 * 4, kernel_size=1, bias=False)
		self.b3 = nn.BatchNorm2d(arg2 * 4)
		self.r = nn.ReLU(inplace=True)
		self.ds = arg4
		self.str = arg3

	def forward(self, x):
		block = x

		o = self.c1(x)
		o = self.b1(o)
		o = self.r(o)

		o = self.c2(o)
		o = self.b2(o)
		o = self.r(o)

		o = self.c3(o)
		o = self.b3(o)

		if self.ds is not None:
			block = self.ds(x)

		o += block
		o = self.r(o)

		return o

class CNN(nn.Module):
	def __init__(self, arg1, arg2, arg3=1000):
		self.ip = 64
		super(CNN, self).__init__()
		self.c1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.b1 = nn.BatchNorm2d(64)
		self.r = nn.ReLU(inplace=True)
		self.mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
		self.apl = self._app(arg1, 64, arg2[0])
		self.ban = self._app(arg1, 128, arg2[1], arg4=2)
		self.oran = self._app(arg1, 256, arg2[2], arg4=2)
		self.pina = self._app(arg1, 512, arg2[3], arg4=2)
		self.tao = nn.AvgPool2d(7)
		self.qr = nn.Linear(512 * arg1.num, arg3)

		for temp in self.modules():
			if isinstance(temp, nn.Conv2d):
				ns = temp.kernel_size[0] * temp.kernel_size[1] * temp.out_channels
				temp.weight.data.normal_(0, math.sqrt(2. / ns))
			elif isinstance(temp, nn.BatchNorm2d):
				temp.weight.data.fill_(1)
				temp.bias.data.zero_()

	def _app(self, arg1, arg2, arg3, arg4=1):
		ds = None
		if arg4 != 1 or self.ip != arg2 * arg1.num:
			ds = nn.Sequential( nn.Conv2d(self.ip, arg2 * arg1.num, kernel_size=1, stride=arg4, bias=False), nn.BatchNorm2d(arg2 * arg1.num) )

		container = []
		container.append(arg1(self.ip, arg2, arg4, ds))
		self.ip = arg2 * arg1.num
		for i in range(1, arg3):
			container.append(arg1(self.ip, arg2))

		return nn.Sequential(*container)

	def forward(self, x):
		x = self.c1(x)
		x = self.b1(x)
		x = self.r(x)
		x = self.mp(x)

		x = self.apl(x)
		x = self.ban(x)
		x = self.oran(x)
		x = self.pina(x)

		x = self.tao(x)
		x = x.view(x.size(0), -1)
		x = self.qr(x)

		return x

class Res(nn.Module):
	def __init__(self, arg1):
		super(Res, self).__init__()
		self.app = arg1

	def forward(self, arg1, arg2=14):
		x = arg1.unsqueeze(0)

		x = self.app.c1(x)
		x = self.app.b1(x)
		x = self.app.r(x)
		x = self.app.mp(x)

		x = self.app.apl(x)
		x = self.app.ban(x)
		x = self.app.oran(x)
		x = self.app.pina(x)

		out1 = x.mean(3).mean(2).squeeze()
		out2 = torch.nn.functional.adaptive_avg_pool2d(x,[arg2,arg2]).squeeze().permute(1, 2, 0)
		
		return out1, out2
