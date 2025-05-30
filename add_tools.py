from langchain_tavily import TavilySearch

tool = TavilySearch(max_results=2)
tools = [tool]
print(tool.invoke("What's a 'node' in LangGraph?"))