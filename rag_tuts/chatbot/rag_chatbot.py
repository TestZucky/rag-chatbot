from langchain_openai import ChatOpenAI
from langchain_community.llms.huggingface_hub import HuggingFaceHub
from langchain_community.document_loaders.pdf import PyPDFLoader #data_ingestion
from langchain_community.vectorstores.faiss import FAISS #vector store
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings #data_ingestion
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent #agent
from langchain.chains.llm import LLMChain # LLM chain
from langchain.memory import ConversationBufferWindowMemory #memory
from langchain.prompts import ChatPromptTemplate #prompt template
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.tools import Tool
from langchain_core.prompts import PromptTemplate
from redisvl.extensions.llmcache import SemanticCache # caching
import time, os, sys
from typing import Tuple


# Get the current directory of the app.py file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Append the parent directory to the system path
sys.path.append(os.path.dirname(current_dir))

from config import Config

os.environ["HUGGINGFACEHUB_API_TOKEN"] = Config.HUGGINGFACEHUB_API_TOKEN # to access huggingface api token
llm_hf = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs={"temperature":0.1, "max_length":512})
llm_openai = ChatOpenAI(openai_api_key=Config.OPENAI_KEY, model='gpt-3.5-turbo-0613') #to access the openai token
embeddings_hf = HuggingFaceEmbeddings()

# Redis semantic cache
llmcache = SemanticCache(
    name="llmcache",
    prefix="llmcache",
    redis_url="redis://localhost:6379",
    distance_threshold=0.1
)

memory = ConversationBufferWindowMemory(memory_key="chat_history", k=3)

# For data ingestion
def load_and_split_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load_and_split()
    return FAISS.from_documents(pages, embeddings_hf)

doc_path = r"gaming_comp.pdf" #your pdf path
db_gaming_comp = load_and_split_pdf(doc_path) # created a vector store
retriever_gaming = db_gaming_comp.as_retriever() #create a retriever

doc_path = r"stellar_data.pdf"
db_stellar_comp = load_and_split_pdf(doc_path) # created a vector store
retriever_stellar = db_stellar_comp.as_retriever() # created a retriever


# for detecting follow up questions
# what is STF? --> false
# tell me its feature --> true
def classify_query(question: str) -> str:
    classification_prompt_template = """
    You are an expert in detecting follow-up questions.
    You will be given a question and you have to detect wether question is follow up or not.
    
    Instructions: A follow-up question is a question that is asked in response to another question, often seeking clarification or more information. 
    You will be given a question, and you have to determine whether it is a follow-up question or not.

    Return True if it is a follow up question otherwise False.

    Input: {question}
    """

    classification_prompt = PromptTemplate(
        template=classification_prompt_template,
        input_variables=["question"],
    )

    llmchain = LLMChain(llm=llm_hf, prompt=classification_prompt, verbose=True)
    res = llmchain.invoke(question)
    return res['text']

# for creating standalone question 
# history --> what is stf?
# new query --> tell its mission  
# stand alone question --> tell the mission of stf?
def create_standalone_question(question: str) -> str:
    standalone_generation_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:
    """

    prompt_for_standalone_question = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=standalone_generation_template
    )

    llm_chain_for_standalone = LLMChain(
        llm=llm_openai,
        prompt=prompt_for_standalone_question,
        verbose=True,
        memory=memory,
    )

    standalone_question = llm_chain_for_standalone.invoke(question)
    return standalone_question['text']

# For cache check
# what is stf
def check_cache(question: str) -> Tuple[str, bool]:
    if response := llmcache.check(prompt=question):
        final_ans = response[0]['response'] # it will bring the response from the cache
        memory.save_context({"input": question}, {"output": final_ans}) # storing it into the history db
        return final_ans, True
    else:
        final_ans = get_answer(question)
        return final_ans, False


# Use agent to answer questions
def get_answer(question: str) -> str:
    
    template_gaming = """You are an assistant having 10 years of experience. You have to act as assistant.
    You will be given context about gaming company data.
    You have to provide concise answer about the questions asked.

    Question: {question}
    Context: {context}
    Answer:
    """
    prompt_gaming = ChatPromptTemplate.from_template(template_gaming)

    template_stellar = """You are an assistant having 10 years of experience. You have to act as assistant.
    You will be given context about Stellar Frontier Technologies company data.
    You have to provide concise answer about the questions asked.

    Question: {question}
    Context: {context}
    Answer:
    """
    prompt_stellar = ChatPromptTemplate.from_template(template_stellar)

    rag_chain_gaming = (
        {"context": retriever_gaming,  "question": RunnablePassthrough()}
        | prompt_gaming
        | llm_openai
        | StrOutputParser()
    )

    rag_chain_stellar = (
        {"context": retriever_stellar,  "question": RunnablePassthrough()}
        | prompt_stellar
        | llm_openai
        | StrOutputParser()
    )

    tools = [
        Tool(
            name="gaming_office_bot",
            func=rag_chain_gaming.invoke,
            description="useful for when you need to answer question related to gaming company(PixelForge Games).",
        ),
        Tool(
            name="stellar_office_bot",
            func=rag_chain_stellar.invoke,
            description="useful for when you need to answer question related to Stellar Frontier Technologies company."
        ),
    ]

    prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
    suffix = """If you dont know the answer just return I dont know the answer. Dont try to make up the answer. \n Begin!

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    
    agent_prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"],
    )

    llm_chain = LLMChain(llm=llm_openai, prompt=agent_prompt)

    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)

    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent, tools=tools, verbose=True, memory=memory
    )

    res = agent_chain.invoke(input=question)
    # print(f"***** Final result: {res} *****")
    llmcache.store(prompt=question, response=res['output']) # store the answer bakc into the cache
    return res['output']

def start_chatbot(user_input: str) -> Tuple[str, bool, float]:
    st_time = time.time()
    # Detect if the question is a follow-up or standalone
    is_follow_up = classify_query(user_input)

    standalone_question = ""

    if is_follow_up == 'True':
        standalone_question = create_standalone_question(user_input)
    else:
        standalone_question = user_input

    # Perform semantic search
    search_result = check_cache(standalone_question) # final_answer
    semantic_search_result = search_result[0] # result
    is_from_cache = search_result[1] # is_in_cache

    end_time = time.time()
    time_taken = end_time - st_time
    return semantic_search_result, is_from_cache, time_taken


# def main():
#     while True:
#         # Get user input
#         user_input = input("Enter your question (or 'exit' to quit): ")

#         # Check if user wants to exit
#         if user_input.lower() == "exit":
#             print("Exiting chatbot...")
#             break
#         start_time = time.time()
#         # Call your chatbot logic
#         response = start_chatbot(user_input)

#         end_time = time.time()

#         return response

# if __name__ == "__main__":
#     main()