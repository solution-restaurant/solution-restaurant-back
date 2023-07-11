# not used Start
from langchain.chains import LLMMathChain
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
# FILE_NAME = "pdf/greating_data_korean_new.csv"
# not used End
import sys
import os
from pathlib import Path
from srback.core.config import settings
from langchain.chat_models import ChatOpenAI
# embedding Start
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
# embedding End
# agent Start 20230709
from langchain.agents import initialize_agent
from langchain.tools import Tool
# agent End
# redisBufferMemory Start
from langchain.memory import ConversationBufferMemory
from langchain.agents import ZeroShotAgent, AgentExecutor
from langchain.memory.chat_message_histories import RedisChatMessageHistory
# redisBufferMemory End
# chain Start
from langchain.chains import RetrievalQAWithSourcesChain
from langchain import LLMChain
# chain End
# prompt Start
from langchain.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
# prompt End





chat_model = ChatOpenAI(
    openai_api_key=os.getenv("OPEN_API_KEY"),
    model_name="gpt-3.5-turbo",
    temperature=0
)
PERSIST_DIRECTORY = 'genChromaDB/db'

def getHealthRecoFood(input):

    persist_directory_path = Path(PERSIST_DIRECTORY).resolve()
    print("PERSIST_DIRECTORY(vectorDB)의 절대 경로:", persist_directory_path)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPEN_API_KEY"))
    vectorDB = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    print(vectorDB)
    
    system_template="""To answer the question at the end, use the following context. If you don't know the answer, say you don't know and don't try to make up an answer.
    I hope you will serve as my great menu recommender. It should only be recommended in the greeting menu.
    Meal plans should include breakfast, lunch and dinner each day. Avoid overlapping as much as possible.
    Calories should be counted only in the main menu
    If you can't make a meal plan, don't say you can't.
    Below is an example.
    You have to plan breakfast, lunch, and dinner and explain the products.
    Recommended product Food names must be clearly stated.
    People with allergies should recommend products other than those for which they are allergic.
    Example: 'Recommended product name' is recommended because of 'food description'.
    
    {summaries}
    """
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain_type_kwargs = {"prompt": prompt}
    bk_chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=chat_model, 
        chain_type="stuff", 
        retriever=vectorDB.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        reduce_k_below_max_tokens=True
    )

    result = bk_chain({"question": input})
    return result['answer']

def getHealthCoachingInfo(input):
    template = '''\
            "{reviews}에 대해서 영양코칭 AI처럼 대답해라"
              \
           '''
    prompt = PromptTemplate(
        input_variables=["reviews"],
        template=template,
    )
    chain = LLMChain(llm=chat_model, prompt=prompt)
    return chain.run(input)

tools = [
    Tool(
        name="영양코칭 AI",
        func=getHealthCoachingInfo,
        description="Use it in a normal conversation",
    ),
    Tool(
        name="상품추천",
        func=getHealthRecoFood,
        description="음식추천과 관련된 내용일 경우 이 툴만 사용해서 답을 줘, '식단추천' 혹은 '식단 추천' 혹은 '상품추천' 혹은 '상품 추천'와 같은 문구가 들어갈 경우에 사용",
        return_direct=True,
    ),
]



# Construct the react agent type.
# health_agent = initialize_agent(
#     # tools,
#     tools=tools,
#     prompt=prompt,
#     llm=chat_model,
#     agent="zero-shot-react-description",
#     verbose=True,
#     # stop=['\nObservation:'],
# )


# new action Start
prefix = """Have a conversation with a human, answering the following questions as best you can. You must have to answer in Korean. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)


# llm_chain = LLMChain(chat_model, prompt=prompt)

# run the agent
message_history = RedisChatMessageHistory(
    url="redis://localhost:6379/0", ttl=600, session_id="my-session"
)

memory = ConversationBufferMemory(
    memory_key="chat_history", chat_memory=message_history
)

llm_chain = LLMChain(llm=chat_model, prompt=prompt)
agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
agent_chain = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, memory=memory
)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        question = ' '.join(sys.argv[1:])
        print('question: ' + question)

        
        # new action End
        # health_agent.run(question)
    else:
        print('agent that answers questions using Weather and Datetime custom tools')
        print('usage: py tools_agent.py <question sentence>')
        print('example: py tools_agent.py what time is it?')