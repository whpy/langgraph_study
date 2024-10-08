# langchain libs
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
# NOTE: you must use langchain-core >= 0.3 with Pydantic v2
from pydantic import BaseModel, Field

# code documents libs
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

# OS libs
import getpass
import os

OA_key = os.environ["OPENAI_API_KEY"]

from typing import List, TypedDict
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        error : Binary flag for control flow to indicate whether test error was tripped
        messages : With user question, error messages, reasoning
        generation : Code solution
        iterations : Number of tries
    """

    error: str
    messages: List
    generation: str
    iterations: int


### OpenAI

# Grader prompt
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{messages}"),
    ]
)
code_fmt_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a coding assistant with expertise in python. You would given a piece of code generated by other llm.\
                  You are responsible to tell the user how to merge the generated code blocks into original code base. The original code base is here:\n{src_code}\
                    here is the generation:\n{generation}""",
        ),
        ("placeholder", "{messages}"),
    ]
)


# Data model
class merged_code(BaseModel):
    """Schema for code solutions to questions about LCEL."""
    file: str = Field(description = "The new code block to be added.")
    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")

class xlblib(BaseModel):
    """ the library of the source code """
    alg: str = Field(description=" the source code of alg.py")
    base: str = Field(description=" the source code of base.py")


fmt_llm = "gpt-4o"

llm_fmt = ChatOpenAI(temperature=0.7, model=fmt_llm)
code_fmt_chain_oai = code_fmt_prompt | llm_fmt.with_structured_output(code)
# | llm_fmt.with_structured_output(code)

f = open("src_code.md")
src_codes = f.read()
f.close()
# req= sys_prompt + "Please help me to implement the solver mentioned in the paper below:\n " + f.read()

f = open("add.md")
r_code = f.read()
print(r_code)
r_fmt = code_fmt_chain_oai.invoke({"src_code":src_codes, "generation":r_code, "messages":[("user",f"hello! please help me to merge the generated code: {r_code}.")]})
print(r_fmt)

def merge(mergement:merged_code):
    file_path = mergement.file
    imports = mergement.imports
    merged_code = mergement.code

    # os.system("cp -r backup/src ")
    f = open(file_path,'a')
    f.write('\n'+ imports + '\n' + merged_code)
    f.close()
    return 

merge(r_fmt)
