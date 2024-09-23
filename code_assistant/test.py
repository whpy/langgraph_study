from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

 # Define a prompt template
prompt_template = PromptTemplate.from_template("Tell me a joke about {topic}")

# Define the input string
input_string = "cats"

# Format the input string into the prompt
formatted_prompt = prompt_template.invoke({"topic": input_string})

# Create a runnable (for example, a chat model)
runnable = ChatOpenAI(model="gpt-3.5-turbo")  # Replace with actual model initialization

# Invoke the runnable with the formatted prompt
response = runnable.invoke(formatted_prompt)

# Print the response
print(response)