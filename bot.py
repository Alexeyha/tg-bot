import logging

import aiohttp

from aiogram import Bot, types
from aiogram.dispatcher import Dispatcher
from aiogram.types import ParseMode

from aiogram.utils.emoji import emojize
from aiogram.utils.executor import start_polling
from aiogram.utils.markdown import bold, code, italic, text
from aiogram.dispatcher.filters.state import State, StatesGroup
from aiogram.dispatcher import FSMContext
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from config import TOKEN
#import my_network
# Configure bot here
API_TOKEN = TOKEN
#PROXY_URL = 'socks5://178.128.203.1:1080'  # Or 'socks5://host:port'

# NOTE: If authentication is required in your proxy then uncomment next line and change login/password for it
#PROXY_AUTH = aiohttp.BasicAuth(login='student', password='TH8FwlMMwWvbJF8FYcq0')
# And add `proxy_auth=PROXY_AUTH` argument in line 30, like this:
# >>> bot = Bot(token=API_TOKEN, proxy=PROXY_URL, proxy_auth=PROXY_AUTH)
# Also you can use Socks5 proxy but you need manually install aiohttp_socks package.

# Get my ip URL
GET_IP_URL = 'http://bot.whatismyipaddress.com/'

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)#, proxy=PROXY_URL, proxy_auth=PROXY_AUTH)

# If auth is required:
# bot = Bot(token=API_TOKEN, proxy=PROXY_URL, proxy_auth=PROXY_AUTH)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
class Choose_photo(StatesGroup):
    content_text = State()
    content_photo = State()
    style_photo = State()

@dp.message_handler(commands=['start', 'help'])
async def send_welcome(message: types.Message):
    """
    This handler will be called when user sends `/start` or `/help` command
    """
    await message.reply("Hi!\nI'm TranStyleBot!\n Send me /style_transfer")

'''
@dp.message_handler()
async def echo(message: types.Message):
    # old style:
    # await bot.send_message(message.chat.id, message.text)

    await message.answer(message.text)
'''
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
    #await Choose_photo.next()
    #my_network
    #import my_network.py
    from my_network import image
    image.save('output.jpg')
    media = types.MediaGroup()
    media.attach_photo(types.InputFile('output.jpg'))
    await message.reply_media_group(media=media)
    await state.finish()

'''@dp.message_handler(content_types=['photo'])
async def handle_docs_photo(message):
    await message.photo[-1].download('test.jpg')
'''

if __name__ == '__main__':
    start_polling(dp, skip_updates=True)