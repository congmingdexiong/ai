from langchain.docstore.wikipedia import Wikipedia
from langchain.llms import OpenAI
from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.agents.react.base import DocstoreExplorer

docstore=DocstoreExplorer(Wikipedia())
tools = [
  Tool(
    name="Search",
    func=docstore.search,
    description="Search for a term in the docstore.",
  ),
  Tool(
    name="Lookup",
    func=docstore.lookup,
    description="Lookup a term in the docstore.",
  )
]

# 使用大语言模型
llm = OpenAI(
  model_name="gpt-3.5-turbo",
  temperature=0,
)

# 初始化ReAct代理
react = initialize_agent(tools, llm, agent="react-docstore", verbose=True)
agent_executor = AgentExecutor.from_agent_and_tools(
  agent=react.agent,
  tools=tools,
  verbose=True,
)


question = "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?"
agent_executor.run(question)