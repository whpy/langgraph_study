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



# LCEL docs
url = "https://python.langchain.com/docs/concepts/#langchain-expression-language-lcel"
loader = RecursiveUrlLoader(
    url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
)
docs = loader.load()

# Sort the list based on the URLs and get the text
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
concatenated_content = "\n\n\n --- \n\n\n".join(
    [doc.page_content for doc in d_reversed]
)


### OpenAI

# Grader prompt
code_gen_prompt = ChatPromptTemplate.from_messages(
    [
        ("placeholder", "{messages}"),
    ]
)


# Data model
class code(BaseModel):
    """Schema for code solutions to questions about LCEL."""

    prefix: str = Field(description="Description of the problem and approach")
    imports: str = Field(description="Code block import statements")
    code: str = Field(description="Code block not including import statements")


expt_llm = "o1-mini"
# expt_llm = "gpt-4o-mini"
llm = ChatOpenAI(temperature=1, model=expt_llm)
code_gen_chain_oai = code_gen_prompt | llm

r = code_gen_chain_oai.invoke({"context": "you are a good assistant.", 
                               "messages": [
                                   ("user",f"hello!")
                                   ]
                                   }
                                   )
print(r)