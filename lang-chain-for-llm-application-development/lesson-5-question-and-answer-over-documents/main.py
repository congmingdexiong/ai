import openai
import os
# ebedding是很多向量的组合，根据不同向量可以确定确定相似度
# 运行时候，用它来查找最相近的片段，每个qry进入时候先为查询创建embeding,然后把它和向量数据库所有向量比较，选择最近的n个
# 并且可以查询与查询相似的文档列表
# 数据库存ebedding,
# 介绍了两种方式根据document查询query，其中一种是传统的调用llm,一种是langchain的方式

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

'''
因为这些文档已经非常小了，所以我们实际上不需要在这里进行任何分块,可以直接进行embedding
'''

from langchain.embeddings import OpenAIEmbeddings #要创建可以直接进行embedding，我们将使用OpenAI的可以直接进行embedding类
embeddings = OpenAIEmbeddings() #初始化

embed = embeddings.embed_query("Hi my name is Harrison")#让我们使用embedding上的查询方法为特定文本创建embedding
print(len(embed))#查看这个embedding，我们可以看到有超过一千个不同的元素
print(embed[:5])#每个元素都是不同的数字值，组合起来，这就创建了这段文本的总体数值表示

loader = CSVLoader(file_path=file,encoding='utf-8')
docs = loader.load()

'''
为刚才的文本创建embedding，准备将它们存储在向量存储中，使用向量存储上的from documents方法来实现。
该方法接受文档列表、嵌入对象，然后我们将创建一个总体向量存储
'''
db = DocArrayInMemorySearch.from_documents(
    docs,
    embeddings
)

query = "Please suggest a shirt with sunblocking"

#使用这个向量存储来查找与传入查询类似的文本，如果我们在向量存储中使用相似性搜索方法并传入一个查询，我们将得到一个文档列表
docs = db.similarity_search(query)

# 我们可以看到它返回了四个文档
print(len(docs))

print(docs[0])

#  如何回答我们文档的相关问题
retriever = db.as_retriever() #创建检索器通用接口
llm = ChatOpenAI(temperature = 0.0,max_tokens=1024) #导入语言模型


##方法一 用query  for in
qdocs = "".join([docs[i].page_content for i in range(len(docs))])  # 将合并文档中的所有页面内容到一个变量中

response = llm.call_as_llm(f"{qdocs} Question: Please list all your \
shirts with sun protection in a table in markdown and summarize each one.") #列出所有具有防晒功能的衬衫并在Markdown表格中总结每个衬衫的语言模型
# print("Please list all your \
# shirts with sun protection in a table in markdown and summarize each one:")
# print(response)

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    verbose=True
)


##方法二  用今天内容实现
query =  "Please list all your shirts with sun protection in a table \
in markdown and summarize each one."#创建一个查询并在此查询上运行链

response = qa_stuff.run(query)

print(response)