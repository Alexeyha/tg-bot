# tg-bot
Весь необходимый код в файле bot.py (при импорте отдельным классом не хватало памяти на aws ??). 

У бота две команды "/style_transfer" и "/gan".

Все фото необходимо отправлять как фото, а не файлом (даже расширением ".jpg" и т.д.)

"/style_transfer" производит перенос стиля, сначала бот предлагает вам прислать контент фото, затем стайл фото. 

"/gan" преобразует фото в стиль комикса (использовалась модель с готовыми весами).
