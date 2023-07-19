import openai
import os

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
# 导入 OpenAI API_KEY
openai.api_key = os.environ['OPENAI_API_KEY']
from langchain.chains import RetrievalQA  # 检索QA链，在文档上进行检索
from langchain.chat_models import ChatOpenAI  # openai模型
from langchain.document_loaders import CSVLoader  # 文档加载器，采用csv格式存储
from langchain.vectorstores import DocArrayInMemorySearch #向量存储
# from IPython.display import display, Markdown  # 在jupyter显示信息的工具

##根据文件内容来回答问题

# 读取文件
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file,encoding='utf-8')

# # 查看数据
# import pandas as pd
# #
# data = pd.read_csv(file, header=None,encoding='utf-8')
# print(data)

from langchain.indexes import VectorstoreIndexCreator

index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])


query ="Please list all your shirts with sun protection \
in a table in markdown and summarize each one."

response = index.query(query)#使用索引查询创建一个响应，并传入这个查询
print(response)

#原理把每句话打散成为向量数组，叫做embedding，
# 有相同内容的文本具有相同的向量
# embedding vector代表内容相同
# vector database：存了被分为不同小块的，为每个小块创建embedding将其村粗，这就是创建索引时候发生的情况
# 运行时候，用它来查找最相近的片段，每个qry进入时候先为查询创建embeding,然后把它和向量数据库所有向量比较，选择最近的n个
# database可以手动传入doc和自己创建的embeding,而且可以返回相似的文档

