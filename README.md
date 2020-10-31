# chatbot_with_bm25

本项目是一个基于bm25搜索算法的对话机器人

数据集是建行小龙人对话机器人的部分日常对话数据，没有任何人工标注

用k-means算法将数据集中的question在句向量空间聚成两类，因为观察数据发现数据中包含业务问题和闲聊，所以希望能通过聚类的方式区分咨询业务问题和闲聊两种意图，以便搜索算法找到最优结果

对聚类后的文本建立倒排索引，记录每个关键词所在的文档id和文档包含关键词的数目

针对用户输入的问题，通过倒排索引找到包含关键词最多的20个相关问题，然后用bm25算法找到最相似的问题并返回答案

通过微信公众号测试最终效果

#### 目录结构:

chatbot.py: 项目的主要类定义

main.py: web服务端程序

handle.py: 处理GET和POST请求的接口

receive.py: 解析从微信公众号平台收到的数据

reply.py: 将回复的数据发给微信公众号

#### 配置&运行:

` git clone git@github.com:chengziyi/chatbot_with_bm25.git                      `

自己训练或下载一个现成的二进制词向量模型放到代码同一目录下命名为'w2v_model.bin'

将数据文件命名为'data.csv'

安装requirements.txt里的依赖

申请并配置好微信公众号，参考：https://developers.weixin.qq.com/doc/offiaccount/Getting_Started/Getting_Started_Guide.html

python main.py 80 运行程序后在公众号测试效果

#### 最终效果：

![image](https://github.com/chengziyi/chatbot_with_bm25/blob/master/images/2.png)
