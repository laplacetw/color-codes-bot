#!usr/bin/env python3
import cv2
import numpy as np
from rec_sys import Analysis
from config import CHANNEL_ACCESS_TOKEN, CHANNEL_SECRET

from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import *


app = Flask(__name__)

# Channel Access Token
line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN)
# Channel Secret
handler = WebhookHandler(CHANNEL_SECRET)

# listen post/request from /callback
@app.route("/callback", methods=['POST'])
def callback():
    # get X-Line-Signature header value
    signature = request.headers['X-Line-Signature']
    # get request body as text
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    # handle webhook body
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

# handle msg
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    if "粉底" in event.message.text:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="請上傳照片讓Pink醬幫您找尋適合的色號！"))
    elif "色號" in event.message.text:
        url = 'https://i.imgur.com/tPC3rBE.jpg'
        line_bot_api.reply_message(
            event.reply_token,
            ImageSendMessage(original_content_url=url,
                             preview_image_url=url))
    else:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="噢！我不清楚你說什麼，請參考功能選單！"))

@handler.add(MessageEvent, message=ImageMessage)
def handle_message(event):
    message_content = line_bot_api.get_message_content(event.message.id)
    nparr = np.fromstring(message_content.content, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    analysis = Analysis()
    img_np = analysis.white_balence(img_np)
    roi_color, eyes = analysis.check(img_np)
    if roi_color is not False:
        result = analysis.anylyze(roi_color, eyes)
        result = 'Pink醬覺得~\n色號 ' + result + ' 很適合你:)'
    else:
        result = '無法辨識，請重新上傳\nლ(•̀ _ •́ ლ)'
    
    line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=result))
    '''
    for chunk in message_content.iter_content():
        print(chunk)
        break
    '''

import os
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
    