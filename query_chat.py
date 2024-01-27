import os
with open("key.txt", "r") as f:
    os.environ["OPENAI_API_KEY"] = f.read()
    
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002"
)

databese = Chroma(
    persist_directory="./.bellmare_gizi",
    embedding_function=embeddings
)

query = "今回のワークショップの目的は？"
documents = databese.similarity_search(query)
document_string = ""
for document in documents:
    document_string += f"""
------------------------------------
{document.page_content}
"""
    
prompt = PromptTemplate(
    template="""
文章を元に質問に答えてください。

文章: 
{document}

質問: {query}
""",
input_variables=["document","query"]
)

chat = ChatOpenAI(
    model="gpt-3.5-turbo"
)

result = chat([
    HumanMessage(content=prompt.format(document=document_string, query= query))
])

print(result.content)