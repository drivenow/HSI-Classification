# -*- coding: utf-8 -*-
import BaseHTTPServer
import urlparse
import time
from keras.models import model_from_json
from PIL import Image
import numpy as np
import threading
import sys
import os

"""
启动服务器，接受url请求的图片名称和地址，返回验证码识别结果
"""
#将one-hot标签变成字符串
#labArr:多行的one-hot编码
def lab2str(labArr):
    num=[]
    strs=""
    for i in labArr:
        for idx,j in enumerate(i):
            if np.round(j)==1:
                num.append(idx)
                break
    for i in num:
        if 0<=i<10:
            strs=strs+str(i)
        if 9<i<37:
            strs=strs+chr(i+97-10)
    return strs
        
        
class HandleHeartBeat:
    def __init__(self,path,query):
        self.path=path
        self.query=query
    def HttpHandle(self):
        return self.query


class HttpRequestHandler(BaseHTTPServer.BaseHTTPRequestHandler):  

   
    def do_GET(self):
        parsed_path = urlparse.urlparse(self.path)
        if (parsed_path.path.startswith("/get"))==False:
            self.wfile.write("request error!"+parsed_path.path)
            return 
        print "================================"
        print "request:"+parsed_path.path
        heartBeat=HandleHeartBeat(parsed_path.path,parsed_path.query.split('&'))
        print heartBeat.HttpHandle()
        self.send_response(200)  
        self.end_headers()

        #读取query
        query=heartBeat.HttpHandle()
        queryDict={}
        print query
        for i in range(len(query)):
            texts=query[i].split("=")
            queryDict[texts[0]]=texts[1]
        #读取验证码图片
        imgPath=queryDict["verfCodePath"]
        imgCtg=queryDict["verfCodeCtg"]
        if  os.path.exists(imgPath)!=True:
            self.wfile.write("verfCodePath Error!")
            return
        
        #图像预处理：
        img=Image.open(imgPath)
        img=np.array(img.convert("L"))
        for i in range(len(img)):
            for j in range(len(img[0])):
                img[i][j]=255-img[i][j]#1.颜色反转，背景黑0，数字为灰度值
                if img[i][j]<70:
                    img[i][j]=0 
        img=np.array([img]).astype('float32')/255.0
        
        #根据编号加载模型
        ctgID={'0':"tuniu"}
        if imgCtg=='0':
            model0=local_school.tuniu_model0   
            model1=local_school.tuniu_model1   
            model2=local_school.tuniu_model2  
            model3=local_school.tuniu_model3   
        
        time.sleep(1)
        pred=[]  
        pred.append(model0.predict(np.array([img]))[0])       
        pred.append(model1.predict(np.array([img]))[0])      
        pred.append(model2.predict(np.array([img]))[0])
        pred.append(model3.predict(np.array([img]))[0])
        
        
        self.wfile.write(lab2str(pred)) 

if __name__== "__main__":    
    #浏览器输入举例：            "http://localhost:8080/getVerfCode?verfCodePath=D:\project\VertCode\tuniu\2000.jpg&verfCodeCtg=0"
    local_school=threading.local()
    modelPath="D:/project/VertCode/mycode/output/model"  
    tuniu_model0 = model_from_json(open(modelPath+"/model0.json").read())    
    tuniu_model0.load_weights(modelPath+"/model0_weight.h5")    
    tuniu_model1 = model_from_json(open(modelPath+"/model1.json").read())    
    tuniu_model1.load_weights(modelPath+"/model1_weight.h5")   
    tuniu_model2 = model_from_json(open(modelPath+"/model2.json").read())    
    tuniu_model2.load_weights(modelPath+"/model2_weight.h5")    
    tuniu_model3 = model_from_json(open(modelPath+"/model3.json").read())    
    tuniu_model3.load_weights(modelPath+"/model3_weight.h5")
    local_school.tuniu_model0=tuniu_model0    
    local_school.tuniu_model1=tuniu_model1   
    local_school.tuniu_model2=tuniu_model2   
    local_school.tuniu_model3=tuniu_model3   
    
            
    server = BaseHTTPServer.HTTPServer(('0.0.0.0',8080), HttpRequestHandler)  
    server.serve_forever()  


#import requests
#url='http://132.228.66.97:8080/heartbeat?a=123&c=123'
#getData=requests.get(url).text
#print getData
