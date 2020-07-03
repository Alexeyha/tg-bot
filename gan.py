import torch

import numpy as np
import wget
from PIL import Image

import torchvision.transforms as transforms
from torch.autograd import Variable

from network import Transformer



valid_ext = ['.jpg', '.png']


# load pretrained model
model = Transformer()
#dict = wget.download('http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Hayao_net_G_float.pth')
dict = 'Hayao_net_G_float.pth'
#dict = wget.download('http://vllab1.ucmerced.edu/~yli62/CartoonGAN/torch_t7/Hayao_net_G_float.t7')
model.load_state_dict(torch.load(dict))# + '_net_G_float.pth')))
torch.save(model, 'model.pth')
model.eval()

model.float()

# load image
input_image = Image.open('test.jpg').convert("RGB")
# resize image, keep aspect ratio
h = input_image.size[0]
w = input_image.size[1]
ratio = h *1.0 / w
if ratio > 1:
	h = 450
	w = int(h*1.0/ratio)
else:
	w = 450
	h = int(w * ratio)
input_image = input_image.resize((h, w), Image.BICUBIC)
input_image = np.asarray(input_image)
	# RGB -> BGR
input_image = input_image[:, :, [2, 1, 0]]
input_image = transforms.ToTensor()(input_image).unsqueeze(0)
	# preprocess, (-1, 1)
input_image = -1 + 2 * input_image

with torch.no_grad():
	input_image = Variable(input_image).float()
# forward
output_image = model(input_image)
output_image = output_image[0]
# BGR -> RGB
output_image = output_image[[2, 1, 0], :, :]
# deprocess, (0, 1)
output_image = output_image.data.cpu().float() * 0.5 + 0.5
output_image = output_image.squeeze(0)  # функция для отрисовки изображения
#def acc():
unloader = transforms.ToPILImage()
output_image = unloader(output_image)
# save
#vutils.save_image(output_image, os.path.join(opt.output_dir, files[:-4] + '_' + opt.style + '.jpg'))
output_image.save('gan.jpg')

