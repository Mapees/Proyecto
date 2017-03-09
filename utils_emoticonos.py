# Librerias
import re
import numpy as np

# ------------------------------
#       EMOTICONOS
# ------------------------------

# Reglas tokenizar
# ICONOS
happy = ":-\)\)|:-\)|:\)|:o\)|:\]|:3|:>|=\]|☺|☻|😁|😂|😄|😅|😃|😊|8\)|=\)|:}|:\^\)|:っ\)|\):-\)|:'-\)|\(:-\)|:'\)"
laugh = ":-D|:D|8-D|8D|x-D|xD|X-D|XD|xd|😆|Xd|=-D|=D|=-3|=3|B^|ahah|eheh|haha"
love = "(♥|♥|❤|♡|♥‿♥|😍|😻|<3<3<3<3|<3<3<3|<3<3|<3)"
annoyed = ">:-\("
sad = ">:\[|\(:-\(|:-\(|:\(|😠|😡|:-Z|☹|:-c|:c|:-<|:っC|:<|:-\[|:\[|:\{|-,-|:-\|\||:@|😞|😐"
cry = ":'-\(|:'\(|QQ|&-l|😓|😥|😭"
disgust = "D:<|:S|:s|😖|D:|D8|D;|D=|DX|v.v|D-':|:\/"
surprise = "(>:O|:-O|:O|°o°|°O°|:O|o_O|o_0|o.O|O.O|O_O|😲|8-0)"
kiss = ":\*|😘|😚|:-\*|:\^\*"
wink = ";-\)|,-\)|;\)|\*-\)|\*\)|;-\]|;\]|;D|;\^\)|:-,|'-\)|\^\^|\^_\^|\^-\^|˘⌣˘|😉|\*_\*|\*-\*"
tongue = ">:P|:-P|:P|X-P|x-p|xp|XP|😋|😝|😜|:-p|:p|=p|:-Þ|:Þ|:-b|:b"
skeptical = ">://|>:\/|:-\/|:-.|:\/|:\\|=\/|:S|>.<|:-7|-_-'|-_-"
indecision = ":-\||\(\(\+_\+\)\)|\(\+o\+\)"
embarassed = ":\$|\(-_-;\)|:-}"
evil = ">:\)|😈|>;\)|>:-\)|}:-\)|}:\)|3:-\)|3:\)"

reglas = evil + "|" + skeptical + "|" +  happy + "|" + laugh + "|" + love + "|" + annoyed + "|" + sad + "|" + cry + "|" + disgust + "|" + surprise + "|" + kiss + "|" + wink + "|" + embarassed + "|" + tongue + "|" +  indecision
    

def tokenize(line):
    result = re.compile( reglas , re.U | re.I)
    matches = re.finditer(result, line)
    return matches
    
def obtener_diccionario_emoticonos(data):
    vector = {}
    for line in data:
        matches = tokenize(line)
        for token in matches:
            symbol = (line[token.start():token.end()]);
            if(symbol not in vector):
                vector[symbol] = 0
    return vector
    
def obtener_vector(data, vector_inicial):
    vector_resultante = []
    vector_resultante_normalizado = []
    for line in data:
        count_tmp = vector_inicial.copy()
        sentence = line.split()
        for i in sentence:
            palabra = i.lower()
            if (palabra in count_tmp):
                valor = count_tmp[palabra]
                valor = valor + 1
                count_tmp[palabra] = valor             
        vector=[]
        vector_temp_norm = []
        for key in count_tmp:
            vector.append(count_tmp[key])
            vector_temp_norm.append(count_tmp[key]/ len(sentence))
        vector_resultante.append(vector)
        vector_resultante_normalizado.append(vector_temp_norm)
    return np.array(vector_resultante), np.array(vector_resultante_normalizado)