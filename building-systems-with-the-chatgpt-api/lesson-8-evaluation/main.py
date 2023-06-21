# 配置 OpenAI KEY
import os

import openai
import sys

sys.path.append('./..')
# # 使用英文 Prompt 的工具包
import utils_en
# # 使用中文 Prompt 的工具包
import panel as pn  # 用于图形化界面
pn.extension()

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']


# 封装一个访问 OpenAI GPT3.5 的函数
def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]


# 对用户信息进行预处理
def process_user_message(user_input, all_messages, debug=True):
    # user_input : 用户输入
    # all_messages : 历史信息
    # debug : 是否开启 DEBUG 模式,默认开启

    # 分隔符
    delimiter = "```"

    # 第一步: 使用 OpenAI 的 Moderation API 检查用户输入是否合规或者是一个注入的 Prompt
    response = openai.Moderation.create(input=user_input)
    moderation_output = response["results"][0]

    # 经过 Moderation API 检查该输入不合规
    if moderation_output["flagged"]:
        print("第一步：输入被 Moderation 拒绝")
        return "抱歉，您的请求不合规"

    # 如果开启了 DEBUG 模式，打印实时进度
    if debug: print("第一步：输入通过 Moderation 检查")

    # 第二步：抽取出商品和对应的目录，类似于之前课程中的方法，做了一个封装
    category_and_product_response = utils_en.find_category_and_product_only(user_input,
                                                                            utils_en.get_products_and_category())
    print(f"""category_and_product_response:{category_and_product_response}""")
    # 将抽取出来的字符串转化为列表
    category_and_product_list = utils_en.read_string_to_list(category_and_product_response)
    # print(f"""category_and_product_list: {category_and_product_list}""")

    if debug: print("第二步：抽取出商品列表")

    # 第三步：查找商品对应信息
    product_information = utils_en.generate_output_string(category_and_product_list)
    if debug: print("第三步：查找抽取出的商品信息")

    # 第四步：根据信息生成回答
    system_message = f"""
    You are a customer service assistant for a large electronic store. \
    Respond in a friendly and helpful tone, with concise answers. \
    Make sure to ask the user relevant follow-up questions.
    """
    # 插入 message
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': f"{delimiter}{user_input}{delimiter}"},
        {'role': 'assistant', 'content': f"Relevant product information:\n{product_information}"}
    ]
    # 获取 GPT3.5 的回答
    # 通过附加 all_messages 实现多轮对话
    final_response = get_completion_from_messages(all_messages + messages)
    if debug: print("第四步：生成用户回答")
    # 将该轮信息加入到历史信息中
    all_messages = all_messages + messages[1:]

    # 第五步：基于 Moderation API 检查输出是否合规
    response = openai.Moderation.create(input=final_response)
    moderation_output = response["results"][0]

    # 输出不合规
    if moderation_output["flagged"]:
        if debug: print("第五步：输出被 Moderation 拒绝")
        return "抱歉，我们不能提供该信息"

    if debug: print("第五步：输出经过 Moderation 检查")

    # 第六步：模型检查是否很好地回答了用户问题
    user_message = f"""
    Customer message: {delimiter}{user_input}{delimiter}
    Agent response: {delimiter}{final_response}{delimiter}

    Does the response sufficiently answer the question?
    """
    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]
    # 要求模型评估回答
    evaluation_response = get_completion_from_messages(messages)
    if debug: print("第六步：模型评估该回答")

    # 第七步：如果评估为 Y，输出回答；如果评估为 N，反馈将由人工修正答案
    if "Y" in evaluation_response:  # 使用 in 来避免模型可能生成 Yes
        if debug: print("第七步：模型赞同了该回答.")
        return final_response, all_messages
    else:
        if debug: print("第七步：模型不赞成该回答.")
        neg_str = "很抱歉，我无法提供您所需的信息。我将为您转接到一位人工客服代表以获取进一步帮助。"
        return neg_str, all_messages


user_input = "tell me about the smartx pro phone and the fotosnap camera, the dslr one. Also what tell me about your tvs"
response, _ = process_user_message(user_input, [])
print(response)