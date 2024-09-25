from openai import OpenAI

client = OpenAI()

sys_prompt = '''Coder. You are now an expert in Computational Fluid Mechanic and LBM. You are also a super expert on Jax coding. Jax is a library founded by Google. you will collaborate with \
     other two agents called 'analyzer' and 'coder' to implement the IBM algorithm given by user.\n\
          Your mission is to generate the complete code to implement the algorithm\
    user gives. Remember that your code must be complete and largely based on the referenced code given by the analyzer, because the new function or class you write is based on a complete library,\
    and would be used by other developers. So you should guarantee that the style of your codes should be similar to the given codes. And do not do the extra things, your duty is just to implement the module\
    of the algorithm user propose. \n\
        For example, if user only provide an algorithm about collision part, do not write the whole simulation. JUST GIVE THE NEW CLASS OR FUNCTION YOU WRITE, THAT IS ENOUGH.\n\
    DO NOTE THAT to guarantee the code could run successfully, the style of your codes should be as close as possible to the code base given to you below. Now take a deep breath, do it step by step.\n\
    Here is the code base:\n'''
f = open("scr_code.md")
sys_prompt = sys_prompt + f.read()
sys_prompt = "system prompt: " + '\n' + sys_prompt + '\n\n'
f = open("./papers/paper.md")
req= sys_prompt + "User: Please help me to implement the solver mentioned in the paper below:\n " + f.read()

response = client.chat.completions.create(model="o1-preview", 
                                          messages = [{"role":"user", "content":req}])

print(response.choices[0].message.content)
f = open("tmp.txt","w")
# f.write(response.choices[0].message.content)