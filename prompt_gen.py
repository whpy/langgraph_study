from typing import List

from langchain_core.messages import SystemMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI

template = '''Your job is to get information from a user about what type of prompt template they want to create.

You should get the following information from them:

- What the objective of the prompt is
- What variables will be passed into the prompt template
- Any constraints for what the output should NOT do
- Any requirements that the output MUST adhere to

If you are not able to discern this info, ask them to clarify! Do not attempt to wildly guess.

After you are able to discern all the information, call the relevant tool.'''
def get_message_info(messages):
    return [SystemMessage(content=template)] + messages

class PromptInstructions(BaseModel):
    objective:str
    variables:List
    constraints:List[str]
    requirements:List[str]

llm = ChatOpenAI(model="gpt-4", temperature=0.)
llm_with_tool = llm.bind_tools([PromptInstructions])

def info_chain(state):
    messages = get_message_info(state['messages'])
    response = llm_with_tool.invoke(messages)
    return {"messages": [response]}

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

prompt_system = """Based on the following requirements, write a good prompt template:

{reqs}"""

def get_prompt_messages(messages:list):
    tool_call = None
    other_msgs = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            tool_call = m.tool_calls[0]['args']
        elif isinstance(m, ToolMessage):
            continue
        elif tool_call is not None:
            other_msgs.append(m)
    return [SystemMessage(content=prompt_system.format(reqs=tool_call))] + other_msgs

def prompt_gen_chain(state):
    messages = get_prompt_messages(state["messages"])
    response = llm.invoke(messages)
    return {"messages": [response]}

from typing import Literal
from langgraph.graph import END

def get_state(state):
    messages = state["messages"]
    if isinstance(messages[-1], AIMessage) and messages[-1].tool_calls:
        return "add_tool_message"
    elif not isinstance(messages[-1],HumanMessage):
        return END
    return 'info'

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    messages:Annotated[list, add_messages]

memory = MemorySaver()
workflow = StateGraph(State)
workflow.add_node("info", info_chain)
workflow.add_node("prompt", prompt_gen_chain)

@workflow.add_node
def add_tool_message(state:State):
    return {
        'messages': [
            ToolMessage(content='prompt generated!',
                        tool_call_id=state['messages'][-1].tool_calls[0]['id'])
        ]
    }

workflow.add_conditional_edges("info", get_state)
workflow.add_edge('add_tool_message','prompt')
workflow.add_edge("prompt", END)
workflow.add_edge(START, "info")
graph = workflow.compile(checkpointer=memory)
# graph = workflow.compile()

import uuid 

config = {"configurable": {"thread_id": str(uuid.uuid4())}}

while True:
    user = input("User: ")
    if user in {'q', 'Q'}:
        print('AI: Bye! see you next time!')
        break
    output = None
    for output in graph.stream(
        {'messages':[HumanMessage(content=user)]}, config=config, stream_mode='updates'
    ):
        last_message = next(iter(output.values()))['messages'][-1]
        last_message.pretty_print()
    
    if output and 'prompt' in output:
        print("Done!")