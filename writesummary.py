import os
from langgraph.graph import StateGraph
from langchain_openai import OpenAI
from typing import TypedDict, List
from langgraph.checkpoint.memory import MemorySaver  

# Define the state to store messages and summaries
class SummaryState(TypedDict):
    article: str
    summary: List[str]

# Create the state graph
graph = StateGraph(SummaryState)

# Initialize the language model
key = os.environ["OPENAI_API_KEY"]
llm = OpenAI(api_key=key)

# Define a node that summarizes the article
def summarize_article(state: SummaryState):
    article_text = state['article']
    summary = llm.summarize(article_text)  # Assuming the LLM has a summarization function
    return {'summary': [summary]}

# Define a node to save the summary to a file
def save_summary_to_file(state: SummaryState):
    summary_text = '\n'.join(state['summary'])
    with open('summary.txt', 'w') as file:
        file.write(summary_text)
    return state

# Add the summarization node and file-writing node to the graph
graph.add_node("summarize", summarize_article)
graph.add_node("save_to_file", save_summary_to_file)

# Set entry point (summarize the article first)
graph.set_entry_point("summarize")

# After summarizing, write the summary to a file
graph.add_edge("summarize", "save_to_file")

# Set finish point
graph.set_finish_point("save_to_file")

# Initialize memory to persist state between graph runs (optional)
checkpointer = MemorySaver()

# Compile the graph (this step is important)
app = graph.compile(checkpointer=checkpointer)

# Create the state with your article
initial_state = SummaryState(article="Your article text here...", summary=[])

# Execute the compiled graph to generate the summary and save it to a file
final_state = app.invoke(initial_state)

# Print the summary (for verification)
print(final_state['summary'])

# The summary will also be saved to 'summary.txt'

