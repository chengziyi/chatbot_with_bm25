import hashlib
import web
import reply
import receive
from chatbot import bot
import time
agent=bot()

class Handle(object):
    def POST(self):
        try:
            webData = web.data()
            print ("Handle Post webdata is :", webData)
            #后台打日志
            recMsg = receive.parse_xml(webData)
            if isinstance(recMsg, receive.Msg):
                toUser = recMsg.FromUserName
                fromUser = recMsg.ToUserName
                recv_content=recMsg.Content.decode()
                if recMsg.MsgType == 'text':
                    print('recv:',recv_content)
                    ans=agent.search(recv_content)
                    #print('ans:',ans)
                    #content = "公众号test"
                    replyMsg = reply.TextMsg(toUser, fromUser, ans)
                    return replyMsg.send()
                if recMsg.MsgType == 'image':
                    mediaId = recMsg.MediaId
                    replyMsg = reply.ImageMsg(toUser, fromUser, mediaId)
                    return replyMsg.send()
                else:
                    return reply.Msg().send()
            else:
                print ("暂且不处理")
                return reply.Msg().send()
        except Exception as Argment:
            return Argment

    def GET(self):
        try:
            data = web.input()
            if len(data) == 0:
                return "hello, this is handle view"
            signature = data.signature
            timestamp = data.timestamp
            nonce = data.nonce
            echostr = data.echostr
            token = "123456" #请按照公众平台官网\基本配置中信息填写

            token_list = [token, timestamp, nonce]
            token_list.sort()
            sha1 = hashlib.sha1()
            #map(sha1.update, token_list)
            [sha1.update(i.encode('utf-8')) for i in token_list]
            hashcode = sha1.hexdigest()
            print("handle/GET func: hashcode, signature: ", hashcode, signature)
            if hashcode == signature:
                return echostr
            else:
                return ""
        except Exception as Argument:
            return Argument
