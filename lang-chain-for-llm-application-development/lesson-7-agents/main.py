import os
from langchain.llms import OpenAI
from langchain.docstore.wikipedia import Wikipedia
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

import warnings
warnings.filterwarnings("ignore")
#
from langchain.agents.agent_toolkits import create_openapi_agent
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.chat_models import ChatOpenAI

# 参数temperature设置为0.0，从而减少生成答案的随机性。
llm = ChatOpenAI(temperature=0)


tools = load_tools(
    ["llm-math","wikipedia"],
    llm=llm #第一步初始化的模型
)
agent= initialize_agent(
    tools, #第二步加载的工具
    llm=llm,  # 第一步初始化的模型,
    # agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,  #代理类型
    handle_parsing_errors=True, #处理解析错误
# handle_parsing_errors="Check your output and make sure it conforms!",
    verbose = True #输出中间步骤
)

agent.run("What is the 25% of 300?")

question = "Tom M. Mitchell is an American computer scientist \
and the Founders University Professor at Carnegie Mellon University (CMU)\
what book did he write?"
agent.run(question)

from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool

agent = create_python_agent(
    llm,  #使用前面一节已经加载的大语言模型
    tool=PythonREPLTool(), #使用Python交互式环境工具（REPLTool）
    verbose=True #输出中间步骤
)

customer_list = [["Harrison", "Chase"],
                 ["Lang", "Chain"],
                 ["Dolly", "Too"],
                 ["Elle", "Elem"],
                 ["Geoff","Fusion"],
                 ["Trance","Former"],
                 ["Jen","Ayai"]
                ]

# agent.run(f"""Sort these customers by \
# last name and then first name \
# and print the output: {customer_list}""")


from langchain.agents import tool
from datetime import date

@tool
def time(text: str) -> str:
    """Returns todays date, use this for any \
    questions related to knowing todays date. \
    The input should always be an empty string, \
    and this function will always return todays \
    date - any date mathmatics should occur \
    outside this function."""
    return str(date.today())

agent= initialize_agent(
    tools + [time], #将刚刚创建的时间工具加入到已有的工具中
    llm, #初始化的模型
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,  #代理类型
    handle_parsing_errors=True, #处理解析错误
    verbose = True #输出中间步骤
)
# agent("whats the date today?")