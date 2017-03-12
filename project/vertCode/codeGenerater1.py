#coding=utf-8
import random
import string
import sys
import math
from PIL import Image,ImageDraw,ImageFont,ImageFilter



    
#字体的位置，不同版本的系统会有不同
font_path = 'C:/Windows/winsxs/amd64_microsoft-windows-font-truetype-simsun_31bf3856ad364e35_6.1.7600.16385_none_56fe10b1895fd80b/simsun.ttc'
#生成几位数的验证码
number = 4
#生成验证码图片的高度和宽度
size = (60,30)
#背景颜色，默认为白色
bgcolor = (255,255,255)
#字体颜色，默认为蓝色
fontcolor = (0,0,255)
#干扰线颜色。默认为红色
linecolor = (255,0,0)
#是否要加入干扰线
draw_line = True
#加入干扰线条数的上下限
line_number = (1,5)

#用来随机生成一个字符串
def gene_text(number):
    source = list(string.letters)
    #去掉大写
    tmp=[]
    for i in source:
        if 64<ord(i)<91:
            tmp.append(i)
    source=list(set(source).difference(set(tmp)))
        
    for index in range(0,10):
        source.append(str(index))
    return ''.join(random.sample(source,number))#number是生成验证码的位数
#用来绘制干扰线
def gene_line(draw,width,height):
    begin = (random.randint(0, width), random.randint(0, height))
    end = (random.randint(0, width), random.randint(0, height))
    draw.line([begin, end], fill = linecolor)

##生成验证码
#def gene_code(text):
#    rootPath="D:/project/VertCode/genCode1/"
#    width,height = size #宽和高
#    image = Image.new('RGBA',(width,height),bgcolor) #创建图片
#    font = ImageFont.truetype(font_path,25) #验证码的字体
#    draw = ImageDraw.Draw(image)  #创建画笔
#    
#    font_width, font_height = font.getsize(text)
##    draw.text(((width - font_width) / number, (height - font_height) / number),text,\
##            font= font,fill=fontcolor) #填充字符串
#    #绘制重叠
#    draw.text((0, 20),text,\
#    font= font,fill=fontcolor) #填充字符串
#    
#    if draw_line:
#        gene_line(draw,width,height)
##     image = image.transform((width+30,height+10), Image.AFFINE, (1,-0.3,0,-0.1,1,0),Image.BILINEAR)  #创建扭曲
#    image = image.transform((width+20,height+10), Image.AFFINE, (1,-0.3,0,-0.1,1,0),Image.BILINEAR)  #创建扭曲
#    image = image.filter(ImageFilter.EDGE_ENHANCE_MORE) #滤镜，边界加强
#    image.save(rootPath+text+'.png') #保存验证码图片
#    image.show()
#    
#    # 图形扭曲参数
#生成验证码
def gene_code(number):
    rootPath="D:/project/VertCode/genCode1/"
    width,height = size #宽和高
    image = Image.new('RGBA',(width,height),bgcolor) #创建图片
    font = ImageFont.truetype(font_path,25) #验证码的字体
    draw = ImageDraw.Draw(image)  #创建画笔
    
    word=[]
    for i in range(number):
        word.append(gene_text(1))
        font_width, font_height = font.getsize(word[i])
        #获得随机位置
        if i==0:
            start_width=3+random.randint(0,4)
            start_height=random.randint(1,3)
        else:
            start_width=random.randint(0,4)+font_width-5+start_width
            start_height=random.randint(1,3)
        draw.text((start_width, start_height),word[i],\
            font= font,fill=fontcolor) #填充字符串
    
    if draw_line:
        gene_line(draw,width,height)
    #     image = image.transform((width+30,height+10), Image.AFFINE, (1,-0.3,0,-0.1,1,0),Image.BILINEAR)  #创建扭曲
    image = image.transform((width+20,height+10), Image.AFFINE, (1,-0.3,0,-0.1,1,0),Image.BILINEAR)  #创建扭曲
    image = image.filter(ImageFilter.EDGE_ENHANCE_MORE) #滤镜，边界加强
    image.save(rootPath+"".join(word)+'.png') #保存验证码图片

    

  
    
if __name__ == "__main__":
    for i in range(50):
#        text = gene_text(0) #生成字符串
#        gene_code(text)
        text = gene_text(1) #生成字符串
        gene_code(4)