# gpt.py

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()



# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please answer the user's question based on the provided context."),
    ("user", "Context: {context}\n\nQuestion: {question}")
])

# GPT-4o-mini model from OpenAI
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
output_parser = StrOutputParser()

# Chain
chain = prompt | llm | output_parser

# Function to get answer
def get_answer(question, context_chunks):
    context = "\n\n".join([doc.page_content for doc in context_chunks])
    return chain.invoke({"context": context, "question": question})
