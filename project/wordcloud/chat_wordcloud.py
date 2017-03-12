# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 09:26:09 2017
@author: Administrator
"""
import jieba    #分词包
import numpy    #numpy计算包
import codecs   #codecs提供的open方法来指定打开的文件的语言编码，它会在读取的时候自动转换为内部unicode 
import pandas   
import matplotlib.pyplot as plt
import re

from wordcloud import WordCloud#词云包

#%% 导入大话西游txt文件，分词
filter_re=re.compile("\w|\d")
basePath=u"E:/360Downloads/word_cloud_Valentines_Day/"
f=open(basePath+u"小机灵.txt",'r')
content=f.readlines()
def getContent(content):
    new_content=[]
    for line in content:
        line=line[126:]
        line=line.strip()
        line=re.sub(filter_re,"",line)
        new_content.append(line)
    return new_content
    
content=getContent(content)
f.close()
segment=[]
for line in content:
    segs=jieba.cut(line)
    for seg in segs:
        if len(seg)>1 and seg!='\r\n':
            segment.append(seg)
        
#%% 去停用词（“多喝热水”和闹嘴的小细节可以在这里抹去）
words_df=pandas.DataFrame({'segment':segment})
words_df.head()
stopwords=pandas.read_csv(basePath+"stopwords.txt",index_col=False,quoting=3,sep="\t",names=['stopword'],encoding="utf8")
words_df=words_df[~words_df.segment.isin(stopwords.stopword)]

#%% 统计词频（情侣对话日常高频词）
words_stat=words_df.groupby(by=['segment'])['segment'].agg({"计数":numpy.size})
words_stat=words_stat.reset_index().sort(columns="计数",ascending=False)
words_stat
#%% 做词云（开启亮瞎眼么么哒模式）
wordcloud=WordCloud(font_path=basePath+"simhei.ttf",background_color="black")
wordcloud=wordcloud.fit_words(words_stat.head(1000).itertuples(index=False))
plt.imshow(wordcloud)
plt.show()
#%% 6.自定义背景图做词云
from scipy.misc import imread
import matplotlib.pyplot as plt
from wordcloud import WordCloud,ImageColorGenerator

bimg=imread(basePath+'heart.jpeg')
wordcloud=WordCloud(background_color="white",mask=bimg,font_path=basePath+'simhei.ttf')
wordcloud=wordcloud.fit_words(words_stat.head(4000).itertuples(index=False))
bimgColors=ImageColorGenerator(bimg)
plt.axis("off")
plt.imshow(wordcloud.recolor(color_func=bimgColors))
plt.show()