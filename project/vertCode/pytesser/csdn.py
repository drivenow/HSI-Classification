#coding:utf-8
r"""
    模式：csdn输错两次密码才会出现验证码
    pytesser:添加环境变量“\”,
    image:读取文件“/”
    验证码图片在img文件夹
    windows Error2:pytesser.exe未添加到环境变量（目前不能再spyder成功运行image_2_string）
    'Upgrade-Insecure-Requests'不能为int格式
    pydessetact可以用pip安装,pytesser不可以，而是替换tessdata文件夹
    pytesseract配置exe程序,tesseract改成pytesseract(但是还没成功。。。)，目前的pydessert还是pip安装的，但是引用pytesser_v0.0.1中的tesseract.exe
    最终方案，首先安装tesseractt-ocr.exe,然后pytesseract中的command还是改成tesseract。此时dos能运行，在spyder运行需要在command中添加完整路径
    ptrhon 3.5，dict没有has_key字段，print需要括号
    python 3.5 字符集中文默认就能输出,encode("utf8")将中文编程\x输出
    持续输出验证码错误，直到浏览器标识改了之后  
    python3.5遇到字符串中的\u会读成8进制，应写成r""
    xpath: // 双斜杠 定位根节点，会对全文进行扫描，在文档中选取所有符合条件的内容，以列表的形式返回
      /text() 获取当前路径下的文本内容 
      /@xxxx 提取当前路径下标签的属性值 
"""


import sys
#sys.path.append('C:\Python27\Lib\site-packages\pytesser_v0.0.1\syspytesser.py')
#sys.path.append('C:\Python27\Lib\site-packages\pytesser_v0.0.1')
#sys.path.append('C:\Python27\Lib\site-packages\pytesser_v0.0.1\tesseract.exe')
import time
import urllib
import shutil
import pytesseract 
#import pytesseract
import requests
from PIL import Image

from lxml import etree

config = {'gid': 1}

def parse(s, html, idx):
    result = {}

    tree = etree.HTML(html)
    try:
        result['lt'] = tree.xpath('//input[@name="lt"]/@value')[0]
        result['execution'] = tree.xpath('//input[@name="execution"]/@value')[0]
        result['path'] = tree.xpath('//form[@id="fm1"]/@action')[0]
    except IndexError as e:
        return None
    print ("result:",result)

    valimg = None
    valimgs = tree.xpath('//img[@id="yanzheng"]/@src')
    if len(valimgs) > 0:
        valimg = valimgs[0]
    
    validateCode = None
    if valimg:
        print ("valimg:\n",valimgs)
        fname = 'img/' + str(idx) + '_' + str(config['gid']) + '.jpg'
        config['gid'] = config['gid'] + 1
        ri = s.get("https://passport.csdn.net" + valimg)
#        保存验证码到图片
        with open(fname, 'wb') as f:
            for chk in ri:
                f.write(chk)
            f.close()
        validateCode = pytesseract.image_to_string(Image.open(fname))
#        validateCode = pytesseract.image_to_string(Image.open(fname))
        validateCode = validateCode.strip()
        validateCode = validateCode.replace(' ', '')
        validateCode = validateCode.replace('\n', '')
        result['validateCode'] = validateCode

    return result

def login(usr, pwd, idx):
    s = requests.Session()

    r = s.get('https://passport.csdn.net/account/login',
    headers={'User-Agent':'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.94 Safari/537.36', 'Host': 'passport.csdn.net', })    

    
    while True:
        res = parse(s, r.text, idx)
        if res == None:
            return False
        url = 'https://passport.csdn.net' + res['path']
        form = {'username': usr, 'password':pwd, '_eventId':'submit', 'execution':res['execution'], 'lt':res['lt'],}
        if 'validateCode' in res:
            print u"验证码为:",res['validateCode']
            form['validateCode'] = res['validateCode']
        s.headers.update({
            'User-Agent':'Mozilla/5.0 (Windows NT 6.2; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/27.0.1453.94 Safari/537.36',
            'Accept-Language': 'zh-CN,zh;q=0.8,en-US;q=0.6,en;q=0.4',
            'Content-Type': 'application/x-www-form-urlencoded',
            'Host': 'passport.csdn.net',
            'Origin': 'https://passport.csdn.net',
            'Referer': 'https://passport.csdn.net/account/login',
            'Upgrade-Insecure-Requests': '1',
            })
        r = s.post(url, data=form)

        tree = etree.HTML(r.text)
        err_strs = tree.xpath('//span[@id="error-message"]/text()')
        if len(err_strs) == 0:
            return True
        err_str = err_strs[0]
        print (err_str)
        err =err_str

        validate_code_err = u'验证码错误'
        usr_pass_err = u'帐户名或登录密码不正确，请重新输入'
        try_later_err = u'登录失败连续超过5次，请10分钟后再试'

        if err[:5] == validate_code_err[:5]:
            pass
        elif err[:5] == usr_pass_err[:5]:
            return False
        elif err[:5] == try_later_err[:5]:
            return False
        else:
            return True

if __name__ == '__main__':
#    login(sys.argv[1], sys.argv[2], 0)
    print (login("13770750257@163.com", "s2393701", 0))