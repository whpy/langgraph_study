from llama_cpp import Llama

from llama_cpp import Llama

mpath = "llama-3-neural-chat-v1-8b-Q8_0.gguf"
llm = Llama(
      model_path="./models/"+mpath,
      n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      # n_ctx=2048, # Uncomment to increase the context window
)

# prompt = "Q: Name the planets in the solar system?\nA: "
# history = "" 
# history = history+prompt
# output = llm(
#       history, # Prompt
#       max_tokens=256, # Generate up to 32 tokens, set to None to generate up to the end of the context window
#       stop=["Q:","\n"], # Stop generating just before the model would generate a new question
#       echo=True # Echo the prompt back in the output
# ) # Generate a completion, can also call create_completion
# print(output['choices'][0]['text'])
# history = output['choices'][0]['text']
# if history[-1]!='\n':
#     history = history+"\n"

# history = history + 'Q: what is your name?\n' + "A: "
# output = llm(
#       history, # Prompt
#       max_tokens=256, # Generate up to 32 tokens, set to None to generate up to the end of the context window
#       stop=["Q:","\n"], # Stop generating just before the model would generate a new question
#       echo=True # Echo the prompt back in the output
# ) # Generate a completion, can also call create_completion
# print(output['choices'][0]['text'])

output = llm.create_chat_completion(
      max_tokens = 4096,
      messages = [
          {"role": "system", "content": "You are an assistant who perfectly describes images."},
          {
              "role": "user",
              "content": "What is your name?"
          },
          {
              "role": "assistant",
              "content": "I am an artificial intelligent model. I don't have a name. You could call me llama >_-"
          },
          {
              "role": "user",
              "content": "Please give me a demo of sudoku game by python."
          }
      ]
)
print(output["choices"][0]["message"]["content"])

# text = ""
# for chunk in output:
#     content = chunk.choices[0].delta.content
#     if content:
#         text += content

# print(text)