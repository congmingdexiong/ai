import openai
import os


from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
# 导入 OpenAI API_KEY
openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.chains import RetrievalQA #检索QA链，在文档上进行检索
from langchain.chat_models import ChatOpenAI #openai模型
from langchain.document_loaders import CSVLoader #文档加载器，采用csv格式存储
from langchain.indexes import VectorstoreIndexCreator #导入向量存储索引创建器
from langchain.vectorstores import DocArrayInMemorySearch #向量存储

#加载数据
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file,encoding='utf-8')
data = loader.load()

#查看数据
import pandas as pd
test_data = pd.read_csv(file,header=None)
print(test_data)

'''
将指定向量存储类,创建完成后，我们将从加载器中调用,通过文档记载器列表加载
'''
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])

#通过指定语言模型、链类型、检索器和我们要打印的详细程度来创建检索QA链
llm = ChatOpenAI(temperature = 0.0)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=index.vectorstore.as_retriever(),
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)
print(qa)

