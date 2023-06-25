# 导入 OpenAI API
import os
import openai
import sys
sys.path.append('./..')
import utils_en
import utils_zh
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

# 用户消息
customer_msg = f"""
tell me about the smartx pro phone and the fotosnap camera, the dslr one.
Also, what TVs or TV related products do you have?"""

# 从问题中抽取商品名
products_by_category = utils_en.get_products_from_query(customer_msg)
# 将商品名转化为列表
category_and_product_list = utils_en.read_string_to_list(products_by_category)
# 查找商品对应的信息
product_info = utils_en.get_mentioned_product_info(category_and_product_list)
# 由信息生成回答
assistant_answer = utils_en.answer_user_msg(user_msg=customer_msg, product_info=product_info)

# 问题、上下文
cust_prod_info = {
    'customer_msg': customer_msg,
    'context': product_info
}


# 使用 GPT API 评估生成的回答
def eval_with_rubric(test_set, assistant_answer):
    cust_msg = test_set['customer_msg']
    context = test_set['context']
    completion = assistant_answer

    # 要求 GPT 作为一个助手评估回答正确性
    system_message = """\
    You are an assistant that evaluates how well the customer service agent \
    answers a user question by looking at the context that the customer service \
    agent is using to generate its response. 
    """

    # 具体指令
    user_message = f"""\
You are evaluating a submitted answer to a question based on the context \
that the agent uses to answer the question.
Here is the data:
    [BEGIN DATA]
    ************
    [Question]: {cust_msg}
    ************
    [Context]: {context}
    ************
    [Submission]: {completion}
    ************
    [END DATA]

Compare the factual content of the submitted answer with the context. \
Ignore any differences in style, grammar, or punctuation.
Answer the following questions:
    - Is the Assistant response based only on the context provided? (Y or N)
    - Does the answer include information that is not provided in the context? (Y or N)
    - Is there any disagreement between the response and the context? (Y or N)
    - Count how many questions the user asked. (output a number)
    - For each question that the user asked, is there a corresponding answer to it?
      Question 1: (Y or N)
      Question 2: (Y or N)
      ...
      Question N: (Y or N)
    - Of the number of questions asked, how many of these questions were addressed by the answer? (output a number)
"""

    messages = [
        {'role': 'system', 'content': system_message},
        {'role': 'user', 'content': user_message}
    ]

    response = get_completion_from_messages(messages)
    return response


evaluation_output = eval_with_rubric(cust_prod_info, assistant_answer)
print(evaluation_output)