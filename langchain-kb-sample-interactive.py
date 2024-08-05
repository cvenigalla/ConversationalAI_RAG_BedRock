#import
import os
import numpy as np
import pandas as pd
import streamlit as st
import time
import random
import json
import time
import random
import requests
import boto3
from botocore.exceptions import ClientError

from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT

from langchain_community.retrievers import AmazonKnowledgeBasesRetriever
from langchain_community.retrievers import AmazonKendraRetriever
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.llms import Bedrock

from langchain.memory import ConversationBufferMemory 

from langchain.prompts.prompt import PromptTemplate
from langchain.chains import create_history_aware_retriever

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv
# Load environment variables from .env file
load_dotenv(dotenv_path="/Users/cvenigalla/Documents/llm_models/.env")

# AWS_BEDROCK_KNOWLEDGE_ID from the environment variable
AWS_BEDROCK_KNOWLEDGE_ID = os.getenv('AWS_BEDROCK_KNOWLEDGE_ID')
if not AWS_BEDROCK_KNOWLEDGE_ID:
    raise ValueError("AWS_BEDROCK_KNOWLEDGE_ID not found in environment variables")

# Retrieve the GROQ API token from the environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY API token not found in environment variables")

# Retrieve the GROQ API token from the environment variable
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY API token not found in environment variables")


# Retrieve the QDRANT_API_KEY from the environment variable
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
if not QDRANT_API_KEY:
    raise ValueError("QDRANT_API_KEY API token not found in environment variables")


PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY API token not found in environment variables")


# Retrieve the API token from the environment variable
HF_TOKEN = os.getenv('HUGGINGFACE_HUB_TOKEN')
if not HF_TOKEN:
    raise ValueError("Hugging Face API token not found in environment variables")

# Retrieve the API token from the environment variable
LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')
if not LLAMA_CLOUD_API_KEY:
    raise ValueError("Llama Cloud API token not found in environment variables")

# Config Streamlit app
st.set_page_config(page_title="AI Model review Bot", page_icon=None, layout="centered", initial_sidebar_state="auto", menu_items=None)
st.title("AI Model review Bot")

session = boto3.Session(profile_name="bedrock-user")
# BEDROCK_CLIENT = boto3.client("bedrock-runtime", 'us-east-1')
BEDROCK_CLIENT = boto3.client("bedrock", 'us-east-1')

#Define the retriever
retriever = AmazonKnowledgeBasesRetriever(
    knowledge_base_id=AWS_BEDROCK_KNOWLEDGE_ID,
    credentials_profile_name="bedrock-user",
    client=BEDROCK_CLIENT, 
    retrieval_config={
        "vectorSearchConfiguration": {
            "numberOfResults": 4
        }
    },
)
# Define model parameters
model_params = {'max_tokens_to_sample':2000, 
                "temperature":0.6,
                "top_k":250,
                "top_p":1,
                # "stop_sequences": ["\n\nHuman"]
               }
#Configure LLM anthropic.claude-instant-v1
# llm = Bedrock(
#   model_id="amazon.titan-text-lite-v1",
#   credentials_profile_name="bedrock-user",
#   client=BEDROCK_CLIENT
# )

llm = Bedrock(model_id = "anthropic.claude-instant-v1",
                    model_kwargs = model_params,
                    credentials_profile_name="bedrock-user",
                    client=BEDROCK_CLIENT
                  )

# Set up message history
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(
    chat_memory = msgs,
    memory_key = 'chat_history',
    output_key= 'answer',
    ai_prefix = "Human: ",
    human_prefix = "AI: ",
    memory_size = 100,
    memory_key_to_save = 'history' # key to save the memory to the streamlit session 
)
# Set up prompt
template = """Use the following peice of context to answer the question at the end.
if you do not find the answer, say i do not understand.
{context}
{chat_history}
Question: {question}
Helpful Answer: {answer}
"""

# Config the template
prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question", "answer"],
    template=template
)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

chat_history = []

rag_chain = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Configure the chain or retriever
# retriever = RetrievalQA.from_chain_type(
#     llm = llm,
#     retriever = retriever,
#     return_source_documents = True,
#     chain_type_kwargs = {"prompt" : template}
# )

# retriever = AmazonKendraRetriever(
#     index_id='Add_IndexId',
#     top_k='0.5'
# )

# # Render current messages from StreamlitChatMessageHistory
for msg in msgs:
    st.chat_message(msg.type).write(msg.content)

# # If user inputs the question
# if prompt := st.chat_input():
#     st.chat_message('human').write(prompt)

#     # Invoke the LLM
#     output = retriever.invoke({'query': prompt})

#     # write into the memory
#     memory.chat_memory.add_user_message(prompt)
#     memory.chat_memory.add_ai_message(output['result'])

#     # display the AI response
#     st.chat_message("ai").write(output['result'])

# Get user input and display the result
while True:
    query = input("\n Ask a Question\n")

    # invoke the model
    output = rag_chain.invoke({"input": query, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=query), output["answer"]])

    # output = retriever.invoke(query)

    # Display the result
    print(output['result'])