import llama_cpp
import llama_cpp.llama_tokenizer

import gradio as gr

# llama = llama_cpp.Llama.from_pretrained(
#     repo_id="Qwen/Qwen1.5-0.5B-Chat-GGUF",
#     filename="*q8_0.gguf",
#     tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained(
#         "Qwen/Qwen1.5-0.5B"
#     ),
#     verbose=False,
# )

mpath = "deepseek-llm-7b-chat.Q8_0.gguf"
llama = llama_cpp.Llama(
      model_path="../models/"+mpath,
      n_gpu_layers=-1, # Uncomment to use GPU acceleration
      # seed=1337, # Uncomment to set a specific seed
      n_ctx=2048, # Uncomment to increase the context window
)

model = "gpt-3.5-turbo"


def predict(message, history):
    messages = []
    messages.append({"role": "system", 
                     "content": 
                     "  You are a helpful assistant. You would get the input as the format below: \n\
                     [user]:****\n\
                     [system_output]:****\n \
                      contents behind '[user]:' are the input user type in the bash shell; contents behind '[system_output]:' are the output when bash receive the user input.\n\
                        Now here is your task:\n\
                     1. If you judge that the output is too verbose and long (like the --help of some command, a very long output of error.), you need to summarize it;\n\
                     2. If you judge that user inputs some wrong prompt, you need to try to speculate what commands user really wants to input and tell user\n\
                     3. If you find that user inputs the requirements in natural language (definitely system output would be a lot of error), you should try to fulfill the requirements by providing related bash command"})
    for user_message, assistant_message in history:
        messages.append({"role": "user", "content": user_message})
        messages.append({"role": "assistant", "content": assistant_message})

    messages.append({"role": "user", "content": message})

    response = llama.create_chat_completion_openai_v1(
        model=model, messages=messages, stream=True
    )

    text = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            text += content
            yield text


js = """function () {
  gradioURL = window.location.href
  if (!gradioURL.endsWith('?__theme=dark')) {
    window.location.replace(gradioURL + '?__theme=dark');
  }
}"""

css = """
footer {
    visibility: hidden;
}
full-height {
    height: 100%;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), js=js, css=css, fill_height=True) as demo:
    gr.ChatInterface(
        predict,
        fill_height=True,
        examples=[
            "What is the definition of Reynolds number?",
            "Do u know what is nematic active matter?",
        ],
    )


if __name__ == "__main__":
    demo.launch(share=True)