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
# print(test_data)

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
# print('data[10,11]')
# print(data[10])
# print(data[11])

# 方法一
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]
# print(qa.run(examples[0]["query"]))


from langchain.evaluation.qa import QAGenerateChain #导入QA生成链，它将接收文档，并从每个文档中创建一个问题答案对

example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI())#通过传递chat open AI语言模型来创建这个链

new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
) #我们可以创建许多例子
print(new_examples)


examples += new_examples
import langchain
langchain.debug = True
qa.run(examples[0]["query"])

#如何评估有两种 ，一种手动，一种自动

langchain.debug = False
predictions = qa.apply(examples)

''' 
对预测的结果进行评估，导入QA问题回答，评估链，通过语言模型创建此链
'''
from langchain.evaluation.qa import QAEvalChain #导入QA问题回答，评估链

#通过调用chatGPT进行评估
llm = ChatOpenAI(temperature=0)
eval_chain = QAEvalChain.from_llm(llm)

graded_outputs = eval_chain.evaluate(examples, predictions)#在此链上调用evaluate，进行评估

#我们将传入示例和预测，得到一堆分级输出，循环遍历它们打印答案
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()