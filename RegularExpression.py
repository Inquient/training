import re

text = 'Тhis is is a test test'
list_text = text.split()

# Ищет в тексте повторяющиеся слова и выделяет их тегом
result = ''
for word in list_text:
    r = re.findall(word+" "+word, text)
    print(r)
    if(r):
       text = re.sub(r[0], word + " <strong>" + word + "</strong>", text)
       print(text)

s = 'asdass ajdshaj ajsdhakjfh ldsfjriwe bfsvn awiedon asdnakndka auedh'
a = re.findall(r'(raven|grotto)', s)