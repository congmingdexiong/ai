import os
import openai
import tiktoken

from dotenv import load_dotenv, find_dotenv
import json

#通过生成输出作为模型输入的一部分来实现

_ = load_dotenv(find_dotenv())

openai.api_key = os.environ['OPENAI_API_KEY']

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
system_message = f"""
You are an assistant that evaluates whether \
customer service agent responses sufficiently \
answer customer questions, and also validates that \
all the facts the assistant cites from the product \
information are correct.
The product information and user and customer \
service agent messages will be delimited by \
3 backticks, i.e. ```.
Respond with a Y or N character, with no punctuation:
Y - if the output sufficiently answers the question \
AND the response correctly uses product information
N - otherwise

Output a single letter only.
"""
customer_message = f"""
tell me about the smartx pro phone and \
the fotosnap camera, the dslr one. \
Also tell me about your tvs"""

product_information = """{ "name": "SmartX ProPhone", "category": "Smartphones and Accessories", "brand": "SmartX", "model_number": "SX-PP10", "warranty": "1 year", "rating": 4.6, "features": [ "6.1-inch display", "128GB storage", "12MP dual camera", "5G" ], "description": "A powerful smartphone with advanced camera features.", "price": 899.99 } { "name": "FotoSnap DSLR Camera", "category": "Cameras and Camcorders", "brand": "FotoSnap", "model_number": "FS-DSLR200", "warranty": "1 year", "rating": 4.7, "features": [ "24.2MP sensor", "1080p video", "3-inch LCD", "Interchangeable lenses" ], "description": "Capture stunning photos and videos with this versatile DSLR camera.", "price": 599.99 } { "name": "CineView 4K TV", "category": "Televisions and Home Theater Systems", "brand": "CineView", "model_number": "CV-4K55", "warranty": "2 years", "rating": 4.8, "features": [ "55-inch display", "4K resolution", "HDR", "Smart TV" ], "description": "A stunning 4K TV with vibrant colors and smart features.", "price": 599.99 } { "name": "SoundMax Home Theater", "category": "Televisions and Home Theater Systems", "brand": "SoundMax", "model_number": "SM-HT100", "warranty": "1 year", "rating": 4.4, "features": [ "5.1 channel", "1000W output", "Wireless subwoofer", "Bluetooth" ], "description": "A powerful home theater system for an immersive audio experience.", "price": 399.99 } { "name": "CineView 8K TV", "category": "Televisions and Home Theater Systems", "brand": "CineView", "model_number": "CV-8K65", "warranty": "2 years", "rating": 4.9, "features": [ "65-inch display", "8K resolution", "HDR", "Smart TV" ], "description": "Experience the future of television with this stunning 8K TV.", "price": 2999.99 } { "name": "SoundMax Soundbar", "category": "Televisions and Home Theater Systems", "brand": "SoundMax", "model_number": "SM-SB50", "warranty": "1 year", "rating": 4.3, "features": [ "2.1 channel", "300W output", "Wireless subwoofer", "Bluetooth" ], "description": "Upgrade your TV's audio with this sleek and powerful soundbar.", "price": 199.99 } { "name": "CineView OLED TV", "category": "Televisions and Home Theater Systems", "brand": "CineView", "model_number": "CV-OLED55", "warranty": "2 years", "rating": 4.7, "features": [ "55-inch display", "4K resolution", "HDR", "Smart TV" ], "description": "Experience true blacks and vibrant colors with this OLED TV.", "price": 1499.99 }"""

q_a_pair = f"""
Customer message: ```{customer_message}```
Product information: ```{product_information}```
Agent response: ```{final_response_to_customer}```

Does the response use the retrieved information correctly?
Does the response sufficiently answer the question?

Output Y or N
"""
#判断相关性
messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': q_a_pair}
]

response = get_completion_from_messages(messages, max_tokens=1)
print(response)

another_response = "life is like a box of chocolates"
q_a_pair = f"""
Customer message: ```{customer_message}```
Product information: ```{product_information}```
Agent response: ```{another_response}```

Does the response use the retrieved information correctly?
Does the response sufficiently answer the question?

Output Y or N
"""
messages = [
    {'role': 'system', 'content': system_message},
    {'role': 'user', 'content': q_a_pair}
]

response = get_completion_from_messages(messages)
print(response)