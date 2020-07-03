import logging

import aiohttp

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher

from aiogram.utils.executor import start_polling
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from config import TOKEN
API_TOKEN = TOKEN
#PROXY_URL = 'socks5://178.128.203.1:1080'  # Or 'socks5://host:port'



#GET_IP_URL = 'http://bot.whatismyipaddress.com/'

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)

# bot = Bot(token=API_TOKEN, proxy=PROXY_URL, proxy_auth=PROXY_AUTH)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
class Choose_photo(StatesGroup):
    content_text = State()
    content_photo = State()
    style_photo = State()

class Gan_photo(StatesGroup):
    content_text = State()
    content_photo = State()

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    await message.reply("Hi!\nI'm TranStyleBot!\nSend me /style_transfer or /gan :))")


@dp.message_handler(commands=['gan'], state='*')
async def download_style(message: types.Message):
    await message.reply("Give me picture!")
    await Gan_photo.content_text.set()
    await Gan_photo.next()

@dp.message_handler(content_types=['photo'], state=Gan_photo.content_photo)
async def download_content(message: types.Message, state: FSMContext):
    await message.photo[-1].download('test.jpg')
    import torch

    import numpy as np

    from PIL import Image

    import torchvision.transforms as transforms
    from torch.autograd import Variable

    from network import Transformer

    valid_ext = ['.jpg', '.png']


    model = Transformer()
    # dict = wget.download('http://vllab1.ucmerced.edu/~yli62/CartoonGAN/pytorch_pth/Hayao_net_G_float.pth')
    dict = 'Hayao_net_G_float.pth'
    model.load_state_dict(torch.load(dict))  
    #model = torch.load('model.pth')
    model.eval()

    model.float()

    # load image
    input_image = Image.open('test.jpg').convert("RGB")
    # resize image, keep aspect ratio
    h = input_image.size[0]
    w = input_image.size[1]
    ratio = h * 1.0 / w
    if ratio > 1:
        h = 310
        w = int(h * 1.0 / ratio)
    else:
        w = 310
        h = int(w * ratio)
    input_image = input_image.resize((h, w), Image.BICUBIC)
    input_image = np.asarray(input_image)

    input_image = input_image[:, :, [2, 1, 0]]
    input_image = transforms.ToTensor()(input_image).unsqueeze(0)

    input_image = -1 + 2 * input_image

    with torch.no_grad():
        input_image = Variable(input_image).float()
    output_image = model(input_image)
    output_image = output_image[0]
    output_image = output_image[[2, 1, 0], :, :]
    output_image = output_image.data.cpu().float() * 0.5 + 0.5
    output_image = output_image.squeeze(0)  # функция для отрисовки изображения
    unloader = transforms.ToPILImage()
    output_image = unloader(output_image)
    output_image.save('gan.jpg')
    media = types.MediaGroup()
    media.attach_photo(types.InputFile('gan.jpg'))
    await message.reply_media_group(media=media)
    await message.reply("Ready! To continue choose /gan or /style_transfer")
    await state.finish()


@dp.message_handler(commands=['style_transfer'], state="*")
async def send_content(message: types.Message):
    await message.reply("Give me content photo!")
    await Choose_photo.content_text.set()
    await Choose_photo.next()

@dp.message_handler(state=Choose_photo.content_photo, content_types=['photo'])
async def download_content(message: types.Message):
    await message.photo[-1].download('content.jpg')
    await message.reply("Give me style photo!")
    await Choose_photo.next()

@dp.message_handler(state=Choose_photo.style_photo, content_types=['photo'])
async def download_style(message: types.Message, state: FSMContext):
    await message.photo[-1].download('style.jpg')

    # %%

    from PIL import Image

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim

    #import matplotlib.pyplot as plt

    import torchvision.transforms as transforms
    import torchvision.models as models

    import copy
    # %%
    imsize = 128

    loader = transforms.Compose([
        transforms.Resize(imsize),  # нормируем размер изображения
        transforms.CenterCrop(imsize),
        transforms.ToTensor()])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def image_loader(image_name):
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0)
        return image.to(device, torch.float)

    style_img = image_loader("style.jpg")
    content_img = image_loader("content.jpg")

    unloader = transforms.ToPILImage()  # тензор в кратинку

    # %%
    class ContentLoss(nn.Module):

        def __init__(self, target, ):
            super(ContentLoss, self).__init__()
            self.target = target.detach()  
            self.loss = F.mse_loss(self.target, self.target) 

        def forward(self, input):
            self.loss = F.mse_loss(input, self.target)
            return input

    # $$
    def gram_matrix(input):
        batch_size, h, w, f_map_num = input.size()  

        features = input.view(batch_size * h, w * f_map_num)  

        G = torch.mm(features, features.t())  

        return G.div(batch_size * h * w * f_map_num)

    # %%
    class StyleLoss(nn.Module):
        def __init__(self, target_feature):
            super(StyleLoss, self).__init__()
            self.target = gram_matrix(target_feature).detach()
            self.loss = F.mse_loss(self.target, self.target)  # to initialize with something

        def forward(self, input):
            G = gram_matrix(input)
            self.loss = F.mse_loss(G, self.target)
            return input

    # %%
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # %%

    class Normalization(nn.Module):
        def __init__(self, mean, std):
            super(Normalization, self).__init__()
            # .view the mean and std to make them [C x 1 x 1] so that they can
            # directly work with image Tensor of shape [B x C x H x W].
            # B is batch size. C is number of channels. H is height and W is width.
            self.mean = torch.tensor(mean).view(-1, 1, 1)
            self.std = torch.tensor(std).view(-1, 1, 1)

        def forward(self, img):
            # normalize img
            return (img - self.mean) / self.std

    # %%
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # %%
    def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                                   style_img, content_img,
                                   content_layers=content_layers_default,
                                   style_layers=style_layers_default):
        cnn = copy.deepcopy(cnn)

        normalization = Normalization(normalization_mean, normalization_std).to(device)

        content_losses = []
        style_losses = []

        model = nn.Sequential(normalization)

        i = 0  
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                # add style loss:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)
                
        for i in range(len(model) - 1, -1, -1):
            if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
                break

        model = model[:(i + 1)]

        return model, style_losses, content_losses

    # %%
    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    # %%
    def run_style_transfer(cnn, normalization_mean, normalization_std,
                           content_img, style_img, input_img, num_steps=150,
                           style_weight=100000, content_weight=1):
        model, style_losses, content_losses = get_style_model_and_losses(cnn,
                                                                         normalization_mean, normalization_std,
                                                                         style_img, content_img)
        optimizer = get_input_optimizer(input_img)

        run = [0]
        while run[0] <= num_steps:
            def closure():
                input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                model(input_img)
                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                    # взвешивание ощибки
                style_score *= style_weight
                content_score *= content_weight

                loss = style_score + content_score
                loss.backward()

                run[0] += 1
                return style_score + content_score

            optimizer.step(closure)
        # a last correction...
        input_img.data.clamp_(0, 1)

        return input_img

    # %%
    input_img = content_img.clone()
 
    # %%
    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)
    # %%
    image = output.cpu().clone()
    image = image.squeeze(0)  
    image = unloader(image)
    image.save('output.jpg')
    media = types.MediaGroup()
    media.attach_photo(types.InputFile('output.jpg'))
    await message.reply_media_group(media=media)
    await message.reply("Ready! To continue choose /gan or /style_transfer")
    await state.finish()


if __name__ == '__main__':
    start_polling(dp)
