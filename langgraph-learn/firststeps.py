from langgraph.graph import StateGraph, MessagesState, START, END

def mock_llm(state : MessagesState):
    k = state["messages"][-1].content + "-> LLM1"
    print(k)
    return {"messages" : [{"role" : "ai", "content" : k}]}

def mock_llm2(state : MessagesState):
    k = state["messages"][-1].content + "-> LLM2"
    print(k)
    return {"messages" : [{"role" : "ai", "content" : k}]}

graph = StateGraph(MessagesState)
graph.add_node(mock_llm)
graph.add_node(mock_llm2)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", "mock_llm2")
graph.add_edge("mock_llm", END)
graph.add_edge("mock_llm2", END)
graph.add_edge(START, END)

graph = graph.compile()

result = graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})["messages"][-1].content
print(result)
# Save the graph visualization as a PNG file
with open("graph.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())
