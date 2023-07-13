import openai
import os

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
# 导入 OpenAI API_KEY
openai.api_key = os.environ['OPENAI_API_KEY']


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(temperature=0.0)
memory = ConversationBufferMemory()
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

print(conversation.predict(input="Hi, my name is Andrew"))

print(conversation.predict(input="What is 1+1"))

print(conversation.predict(input="What is my name"))

print(memory.load_memory_variables({}))

memory = ConversationBufferMemory()
memory.save_context({"input": "Hi"}, {"output": "what's up"})
print(memory)
print(memory.load_memory_variables({}))

# 仅仅记住一次 传入这个memory不能记住上下文了
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=1)
memory.save_context({"input": "Hi"}, {"output": "what's up"})
memory.save_context({"input": "Hi1"}, {"output": "what's up1"})
print("仅仅记住一次")
print(memory.load_memory_variables({}))
#  限制token数量，进行测试
from langchain.memory import ConversationTokenBufferMemory

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=30)
memory.save_context({"input": "AI is what?!"},
                    {"output": "Amazing!"})
memory.save_context({"input": "Backpropagation is what?"},
                    {"output": "Beautiful!"})
memory.save_context({"input": "Chatbots are what?"},
                    {"output": "Charming!"})
print("限制token数量，进行测试")
print(memory.load_memory_variables({}))

from langchain.memory import ConversationSummaryBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain

# 把上面对话总结成为一个summary,然后当下次再问这个问题的时候，会提取这个summary作为参考
schedule = "There is a meeting at 8am with your product team. \
You will need your powerpoint presentation prepared. \
9am-12pm have time to work on your LangChain \
project which will go quickly because Langchain is such a powerful tool. \
At Noon, lunch at the italian resturant with a customer who is driving \
from over an hour away to meet you to understand the latest in AI. \
Be sure to bring your laptop to show the latest LLM demo."

memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)  # 使用对话摘要缓存记忆
memory.save_context({"input": "Hello"}, {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "What is on the schedule today?"},
                    {"output": f"{schedule}"})
print("ConversationSummaryBufferMemory:")
print(memory.load_memory_variables({}))

conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)
print(conversation.predict(input="What would be a good demo to show?"))
print(memory.load_memory_variables({}))
