import sys
import os
from langchain.chains import LLMMathChain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferMemory
import os

from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


from pathlib import Path



# import custom tools
from srback.core.config import settings
from langchain.tools import Tool
llm = ChatOpenAI(openai_api_key=os.getenv("OPEN_API_KEY"), model_name='gpt-3.5-turbo')
chat = OpenAI(openai_api_key=os.getenv("OPEN_API_KEY"), model_name='gpt-3.5-turbo')

# FILE_NAME = "pdf/greating_data_korean_new.csv"
PERSIST_DIRECTORY = 'genChromaDB/db'

def getHealthRecoFood(input):

    persist_directory_path = Path(PERSIST_DIRECTORY).resolve()
    print("PERSIST_DIRECTORY(vectorDB)의 절대 경로:", persist_directory_path)

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPEN_API_KEY"))
    vectorDB = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    print(vectorDB)
    
    system_template="""끝에 있는 질문에 답하려면 다음 컨텍스트를 사용하십시오. 답을 모르면 모른다고 말하고 답을 지어내려고 하지 마세요.
    당신이 나의 훌륭한 메뉴 추천인 역할을 해주기를 바랍니다. 그리팅 메뉴에서만 추천해야 한다.
    식사 계획은 매일 아침, 점심, 저녁 식사를 포함해야 합니다. 최대한 겹치지 않도록 합니다.
    칼로리는 대메뉴 값만 따져봐야
    식사 계획을 세울 수 없다면 할 수 없다고 말하지 마십시오.
    아래는 예시입니다.
    아침, 점심, 저녁 식단 짜주고 상품 설명 해줘야합니다.
    추천 상품 음식 이름은 정확이 명시해 줘야합니다. 
    알러지가 있는 사람은 알러지 해당하는 제품을 제외한 나머지 제품중 추천 해줘야합니다.
    예시 : '추천상품 명'을 추천합니다 그 이유는'음식 설명'때문입니다.
    {summaries}
    """
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain_type_kwargs = {"prompt": prompt}
    bk_chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(openai_api_key=os.getenv("OPEN_API_KEY"),  model_name="gpt-3.5-turbo", temperature=0), 
        chain_type="stuff", 
        retriever=vectorDB.as_retriever(),
        chain_type_kwargs=chain_type_kwargs,
        reduce_k_below_max_tokens=True
    )

    result = bk_chain({"question": input})
    # result = bk_chain({"question": '300kcal 이하의 최종금액이 10000원이하 아침, 점심, 저녁 식단 짜주고 상품 설명 해줘'}) result['answer']
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
    chain = LLMChain(llm=chat, prompt=prompt)
    return chain.run(input)

tools = [
    Tool(
        name="영양코칭 AI",
        func=getHealthCoachingInfo,
        description="영양코칭AI로서 대답해주는 역할.",
    ),
    Tool(
        name="상품추천",
        func=getHealthRecoFood,
        description="음식추천과 관련된 내용일 경우 이 툴만 사용해서 답을 줘, '식단추천' 혹은 '식단 추천' 혹은 '상품추천' 혹은 '상품 추천'와 같은 문구가 들어갈 경우에 사용",
        return_direct=True,
    ),
]

# Construct the react agent type.
health_agent = initialize_agent(
    tools,
    chat,
    agent="zero-shot-react-description",
    verbose=True,
    stop=['\nObservation:'],
)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        question = ' '.join(sys.argv[1:])
        print('question: ' + question)

        # run the agent
        health_agent.run(question)
    else:
        print('agent that answers questions using Weather and Datetime custom tools')
        print('usage: py tools_agent.py <question sentence>')
        print('example: py tools_agent.py what time is it?')