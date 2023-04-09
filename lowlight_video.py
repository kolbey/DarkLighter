import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import time
import DarkLighter_model as model
import numpy as np
from PIL import Image
import time
import cv2

from thop import clever_format
from thop import profile


	
def pre_process(low_frame):
	
	lowlight_frame = (np.asarray(low_frame)/255.0)
	lowlight_frame = torch.from_numpy(lowlight_frame).float()
	lowlight_frame = lowlight_frame.permute(2,0,1)
	lowlight_frame = lowlight_frame.cuda().unsqueeze(0)
	
	return lowlight_frame

# # 增强结果后处理 Version-1
# def post_process(enhanced_frame):
# 	enhanced_frame = torchvision.utils.make_grid(enhanced_frame, nrow=8, padding=2,
# 					      pad_value=0, normalize=False, range=None, scale_each=False)
# 	ndarr = enhanced_frame.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
# 	return ndarr

# # 增强结果后处理 Version-2
def post_process(enhanced_frame):
	enhanced_frame = np.asarray(enhanced_frame.squeeze(0).permute(1,2,0).cpu())
	enhanced_frame = (enhanced_frame * 255) + 0.5
	enhanced_frame = np.clip(enhanced_frame, 0, 255)
	enhanced_frame = np.uint8(enhanced_frame)
	return enhanced_frame




if __name__ == '__main__':
	# # 测算模型的float与Params
	# img = torch.Tensor(1, 3, 400, 600)
	# net = model.enhancer()
	# flops, params = profile(net, inputs=(img, ))
	# flops, params = clever_format([flops, params], "%.3f")
	# print(flops, params)

	video = cv2.VideoCapture(0)
	while True:
		ret, frame = video.read()
		
		lowlight_frame = pre_process(frame)
		
		with torch.no_grad():
			DarkLighter = model.enhancer().cuda()
			DarkLighter.load_state_dict(torch.load('snapshots/Epoch193.pth'))

			start = time.time()
			enhanced_image,_,_ = DarkLighter(lowlight_frame)
			enhanced_image = post_process(enhanced_image)
			end_time = (time.time() - start)
			print(end_time)
			cv2.imshow('original', frame)
			cv2.waitKey(1)
			cv2.imshow('enhanced', enhanced_image)
			cv2.waitKey(1)
