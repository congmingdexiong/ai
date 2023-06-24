import os
import openai
import tiktoken

from dotenv import load_dotenv, find_dotenv
import json

_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']
#利用Moderation来判断
def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0,
                                 max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message["content"]

final_response_to_customer = f"""
SmartX ProPhone有一个6.1英寸的显示屏，128GB存储、1200万像素的双摄像头，以及5G。FotoSnap单反相机有一个2420万像素的传感器，1080p视频，3英寸LCD和 
可更换的镜头。我们有各种电视，包括CineView 4K电视，55英寸显示屏，4K分辨率、HDR，以及智能电视功能。我们也有SoundMax家庭影院系统，具有5.1声道，1000W输出，无线 
重低音扬声器和蓝牙。关于这些产品或我们提供的任何其他产品您是否有任何具体问题？
"""

response = openai.Moderation.create(
    input=final_response_to_customer
)
moderation_output = response["results"][0]
print(moderation_output)

