import re 
import os

toneMap = { "ā": "a1", "á": "a2", "ǎ": "a3", "à": "a4", "ō": "o1", "ó": "o2", "ǒ": "o3", "ò": "o4", "ē": "e1", "é": "e2", "ě": "e3", "è": "e4",
		"ī": "i1", "í": "i2", "ǐ": "i3", "ì": "i4", "ū": "u1", "ú": "u2", "ǔ": "u3", "ù": "u4", "ü": "v0", "ǖ": "v1", "ǘ": "v2", "ǚ": "v3",
		"ǜ": "v4", "ń": "n2", "ň": "n3", "": "m2"
	};

tone = {'#2': 0, '#1': 0, '#4': 0, '#3': 0} # '#1', '#2', '#3', '#4'

cwd = os.getcwd()
suffix = '' if 'pinyin' in cwd else './tacotron/pinyin'

hz = {}
with open(os.path.join(suffix, 'pinyin.txt'), 'r', encoding='utf-8') as f:
    i = 0 
    for line in f:
        if i < 2:
            i += 1
            continue 
        
        line = line.strip()
        line = re.sub(r'\s+', '', line)
        line = line.split(':')[1].split('#')
        word = line[1].strip()
        py = line[0].strip().split(',')
        hz[word] = py


phrase = {}
with open(os.path.join(suffix, 'large_pinyin.txt'), 'r', encoding='utf-8') as f:
    i = 0 
    for line in f:
        if i < 2:
            i += 1
            continue
        
        line = line.strip().split(':')
        pz = line[0].strip()
        py = line[1].strip().split(' ')
    
        phrase[pz[0]] = phrase.get(pz[0], [])
        phrase[pz[0]].append((pz, py))


def int_to_words(astr):

	amap1 = ['', '万', '亿']
	amap2 = ['', '十', '百', '千']
	digit = {'0':'零', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}

	res = ''
	i = len(astr) - 1
	zero_occur = False 
	i = 0 
	flag = True 

	while i < len(astr):
		j = len(astr) - 1 - i 
		if astr[i] == '0':
			zero_occur = True 
		else:
			if zero_occur:
				res = res + '零'
			zero_occur = False 

			if not (astr[i] == '1' and len(astr) == 2 and j % 4 == 1):
				res = res + digit[astr[i]]

			res = res + amap2[j % 4]

		if j % 4 == 0 and j // 4 > 0:

			res = res + amap1[j // 4]
			if flag:
				res = res + '，'

			zero_occur = False 

		i += 1
	
	aastr = astr 
	for i in range(len(astr)-1, -1, -4):
		if i != len(astr) - 1:
			aastr = aastr[:i+1] + ', ' + aastr[i+1:]

	return res 


def digit_to_words(astr):
	digit = {'0':'零', '1': '一', '2': '二', '3': '三', '4': '四', '5': '五', '6': '六', '7': '七', '8': '八', '9': '九'}
	digit['.'] = '点'
	res = ''
	for ch in astr:
		res = res + digit[ch]

	return res 


def float_to_words(astr):
	part1, part2 = astr.split('.')
	res = int_to_words(part1) + '点' + digit_to_words(part2)


def preprocess(text, tone=False):
    if not tone:
        text = re.sub(r'#\d+', '', text)
    
    text = text.lower()
    text = re.sub(r'[）（]', '', text)
    #text = re.sub(r'[0-9]{0,}\.?[0-9]+', '', text)
    text = text.replace('：“', '，').replace('：', '，').replace('”！', '！').replace('”。', '。')
    text = text.replace('……”', '。').replace('……', '。').replace('…。', '。').replace('…”', '。').replace('…', '。').replace('.', '。')
    text = text.replace('”', '').replace('“', '').replace('、', '，').replace('-', '，')
    text = text.replace('—', '，').replace('-', '，').replace('；', '。')
    text = re.sub(r'，[，\s]+', '，', text)
    text = re.sub(r'。[。，\s]+', '。', text)
    text = re.sub(r'，。+', '。', text)

    text = re.sub(r'？[？\s]+', '？', text)
    text = re.sub(r'，？+', '？', text)

    text = re.sub(r'！[！\s]+', '！', text)
    text = re.sub(r'，！+', '！', text)
    text = re.sub('\.+', '。', text)
    text = re.sub(',+', '，', text)
    text = re.sub('!+', '！', text)
    text = re.sub('\?+', '？', text)

    text = re.sub(r'\s+', ' ', text)
    text = text.replace('|', '') 
    text = text.strip()

    '''
    for t in text:
        if not ('\u4e00' <= t <= '\u9fff' or t in ['，', '。', '？', '！'] or t in ['#', '1', '2', '3', '4']):
            print(t, text)
            break 
    '''
    return text 

def split_pyin(pyin):
    if pyin[:2] in ['ch', 'sh', 'zh']:
        return pyin[:2] + ' ' + pyin[2:]
    elif pyin[0] in ['a', 'e', 'o']:
        return pyin
    elif len(pyin) == 2 and pyin[-1].isdigit():
        return pyin
    else:
        return pyin[0] + ' ' + pyin[1:]
    

def tone_to_digit(pyin):
    for i in range(len(pyin)):
        if pyin[i] in toneMap:
            pyin = pyin[:i] + toneMap[pyin[i]][0] + pyin[i+1:] + toneMap[pyin[i]][1]
            break 
    
    pyin = split_pyin(pyin)

    return pyin


def get_pyin(text, tone=False):
    text = preprocess(text, tone)
    print(text)
    res = []
    i = 0
    while i < len(text):
        if text[i] == 'p' and text[i : i+3] == 'pi1':
            res.append(text[i : i+3]) # bug to fix
            #res.append(text[i: i+1])
            #res.append(text[i+1: i+3])
            i += 3
            continue 
        if text[i] == 'b' and text[i : i+3] == 'bi1':
            res.append(text[i : i+3]) # bug to fix
            #res.append(text[i: i+1])
            #res.append(text[i+1:i+3])
            i += 3
            continue

        if text[i] == '#':
            i += 1
            if i < len(text) and text[i] in ['1', '2', '3', '4']:
                res.append('#' + text[i])
                i += 1

            continue 

        if 'a' <= text[i] <= 'z': # for pinyin input, space split # n i3 h hao3
            j = i 
            while i < len(text) and 'a' <= text[i] <= 'z':
                i += 1
            
            if i < len(text) and text[i] in ['1', '2', '3', '4']:
                i += 1

            res.append(text[j:i])
            if text[i] == ' ':
                i += 1
            continue
            
        tmp = '' 
        while i < len(text) and text[i].isdigit():
            tmp = tmp + text[i]
            i += 1
        if len(tmp) > 0:
            words= int_to_words(tmp)
            pyin1, words = get_pyin(words)
            res.extend(pyin1.split(' '))
            continue 
        
        t = text[i]
        if t in phrase:
            flag = 0 
            for item in phrase[t]:
                pz, py = item
                if text[i:i+len(pz)] == pz:
                    for j in range(len(pz)):
                        res.append(tone_to_digit(py[j]))
                    i += len(pz)
                    flag = 1
                    break 
            if flag == 1:
                continue

        if t in hz:
            #res.append(t + tone_to_digit(hz[t][0]))
            res.append(tone_to_digit(hz[t][0]))
        else:
            res.append(t)
        
        i += 1

    return ' '.join(res), text
            
if __name__ == '__main__':
    text = '大写字母用于随机变量，而小写字母用于随机变量的具体值或标量函数。'
    text = '如果打穿地球，那么从一头到另一头会发生什么？'
    pyin, txt = get_pyin(text)
    print(pyin)
                
