# -*- coding:UTF-8 -*-
import urllib2, cookielib
import time

def getchk(number):
    # 创建cookie对象
    cookie = cookielib.LWPCookieJar()
    cookieSupport = urllib2.HTTPCookieProcessor(cookie)
    opener = urllib2.build_opener(cookieSupport, urllib2.HTTPHandler)
    urllib2.install_opener(opener)
    # 首次与教务系统链接获得cookie#
    # 伪装browser
    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Encoding': 'gzip,deflate',
        'Accept-Language': 'zh-CN,zh;q=0.8',
        'User-Agent': 'Mozilla/5.0 (Windows NT 6.2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36'
    }
    req0 = urllib2.Request(
        url='http://mis.teach.ustc.edu.cn',
        headers=headers  # 请求头
    )
    # 捕捉http错误
    try:
        result0 = urllib2.urlopen(req0)
    except urllib2.HTTPError, e:
        print e.code
    # 提取cookie
    getcookie = ['', ]
    for item in cookie:
        getcookie.append(item.name)
        getcookie.append("=")
        getcookie.append(item.value)
        getcookie = "".join(getcookie)

    # 修改headers
    headers["Origin"] = "http://mis.teach.ustc.edu.cn"
    headers["Referer"] = "http://mis.teach.ustc.edu.cn/userinit.do"
    headers["Content-Type"] = "application/x-www-form-urlencoded"
    headers["Cookie"] = getcookie
    for i in range(number):
        req = urllib2.Request(
            url="https://passport.tuniu.com/ajax/captcha/v/"+str(time.time()),
            headers=headers  # 请求头
        )
        response = urllib2.urlopen(req)
        status = response.getcode()
        picData = response.read()
        if status == 200:
            localPic = open(u"F:/硬盘BACK/学习/PYTHON/验证码识别/验证码-途牛/途牛验证码/" + str(i+2002) + ".jpg", "wb")
            localPic.write(picData)
            localPic.close()
        else:
            print "failed to get Check Code "


if __name__ == '__main__':
    getchk(1000)