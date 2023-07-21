# not used Start
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.chains import LLMMathChain
from langchain.chains import ChatVectorDBChain #deprecated
from langchain.chains import ConversationalRetrievalChain
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
import json
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
from langchain import SQLDatabase, SQLDatabaseChain
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
# outputParser Start
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List
from langchain.output_parsers import PydanticOutputParser
# outputParser End

# global data Start
import srback.api.v1.endpoint.userProperties as globalUser

# global data End

chat_model = ChatOpenAI(
    openai_api_key=os.getenv("OPEN_API_KEY"),
    model_name="gpt-3.5-turbo",
    temperature=0
)
chat_FeedBack_model = ChatOpenAI(
    openai_api_key=os.getenv("OPEN_API_KEY"),
    model_name="gpt-3.5-turbo",
    temperature=0.3
)
PERSIST_DIRECTORY = 'genChromaDB/db'
PERSIST_DIRECTORY2 = 'genChromaDB/db2'
PERSIST_DIRECTORY3 = 'genChromaDB/db3'


class FoodInfo(BaseModel):
    # mealForDay: str = Field(description="morning or lunch or dinner Please use it")
    productName: str = Field(description="productName used to answer the user's question.")
    id: str = Field(description="product Id")
    # comment: str = Field(description="comment used to answer the user's question.")

class FoodList(BaseModel):
    foodList: List[FoodInfo] = Field(description="list of Food")
    reviews: str = Field(description="You must have to answer by writing Korean only. why you recommended the product for user's question like a nutrition consultant.")
    # You can add custom validation logic easily with Pydantic.
    # @validator('setup')
    # def question_ends_with_question_mark(cls, field):
    #     if field[-1] != '?':
    #         raise ValueError("Badly formed question!")
    #     return field
class RecoFoodInfo(BaseModel):
    input: str = Field(description="Tell me more about user's allergies, diseases, interests, and high and low calories user's want")


def getHealthRecoFoodSql(input):
    # db 설정
    db = SQLDatabase.from_uri(os.getenv("DB_PATH")
                            ,include_tables=['meal']
                            ,sample_rows_in_table_info=2)
    
    print("\nglobalUserName : " + globalUser.globalUserName)
    print("allergy : " + globalUser.globalUserAllegy)
    print("diease : " + globalUser.globalUserDisease)
    
    # 유저 기본 값 문구 세팅
    if(input == 'None'):
        print("기본 추천")
        allergy = find_name(globalUser.globalUserAllegy, swapped_allergy_list)
        disease = find_name(globalUser.globalUserDisease, swapped_disease_list)
        if(allergy == 'NONE' and disease == 'NONE'):
            input = "기본 식습관 정보가 저장되어 있지 않아. 무작위로 상품을 추천 해 보았습니다."
        else:
            input = '고객님의 기본 식습관 정보를 바탕으로 '
            if(allergy != 'NONE'):
                if(disease != 'NONE'):
                    input += allergy + "가 없고 " + disease + "에 좋은 상품"
                else:
                    input += allergy + "가 없는 상품 "
              
            elif(disease != 'NONE'):
                input += disease + "에 좋은 상품"
            input += "을 추천 해 보았습니다."
    
    print(input)
    # db 설정

    # template
    template = """Given an input question, first create a syntactically correct MySQL query to run , ORDER BY must appear before the LIMIT clause , then look at the results of the query and return the answer.
    Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the input or RAND() in the database.

    Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

    Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

    Question: "Question here"
    SQLQuery: "SELECT product_name, id"
    SQLResult: "Result of the SQLQuery"
    Answer: "Final answer here"
    
    Question: "땅콩 알러지가 없는 상품에 대해서 추천해줘"
    SQLQuery: SELECT product_name, id WHERE peanut_allergy = 'N'
    
    Question: "당뇨인 사람에게 상품 추천해줘"
    SQLQuery: SELECT product_name, id FROM meal WHERE good_for_diabetes = 'Y'
    
    Question: "나는 우유알러지가 있어 식단 추천해줘"
    SQLQuery: SELECT product_name, id FROM meal WHERE milk_allergy = 'N'
    
    Question: "복숭아 알러지 없고 암에 좋은 상품으로 추천 해줘"
    SQLQuery: SELECT product_name, id FROM meal WHERE peach_allergy = 'N' AND good_for_cancer = 'Y'
    
    Question: "매운 음식 추천 해줘"
    SQLQuery: SELECT product_name, id FROM meal WHERE comment like '%매운%' 
    
    답변 형식은 아래 형식으로 출력해줘.
    you only answer in korean.
    {format_instructions}
    
    
    Only use the following tables:
    {table_info}

    If someone asks for the table product or meals, they really mean the meal table.
    Question: {input}
    """
    # outputParser
    parser = PydanticOutputParser(pydantic_object=FoodList)    
    # response_schemas = [
    #     ResponseSchema(name="content", description="answer to the user's question."),
    #     ResponseSchema(name="product_name", description="product_name used to answer the user's question."),
    #     ResponseSchema(name="comment", description="comment used to answer the user's question.")
    # ]
    # output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    # format_instructions = output_parser.get_format_instructions()

    prompt = PromptTemplate(
        input_variables=["input", "table_info", "top_k"],
        template=template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    # template
    
    db_chain = SQLDatabaseChain.from_llm(
        llm=chat_model, db=db, prompt=prompt, verbose=True, top_k=3
    )
    result = db_chain.run(input)
    
    return result


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

def getNutrientInfo(question):
    persist_directory_path3 = Path(PERSIST_DIRECTORY3).resolve()
    print("PERSIST_DIRECTORY(vectorDB2)의 절대 경로:", persist_directory_path3)

    embeddings3 = OpenAIEmbeddings(openai_api_key=os.getenv("OPEN_API_KEY"))
    vectorDB3 = Chroma(persist_directory=PERSIST_DIRECTORY3, embedding_function=embeddings3)
    print(vectorDB3)

    template = """
        Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't make up an answer. 
    If you don't have the accurate answer from document, just say that you don't know, don't make up an answer.  
    Use three sentences maximum and keep the answer as concise as possible. 
    You must have to answer in Korean.
    {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"],template=template,)

    retriever = vectorDB3.as_retriever(search_type="similarity", search_kwargs={"k": 2})
    # create a chain to answer questions 
    llm = OpenAI(openai_api_key=os.getenv("OPEN_API_KEY"), model_name='gpt-3.5-turbo')
    qa_chain = RetrievalQA.from_chain_type(
        llm=chat_model, chain_type="stuff", retriever=retriever, return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})

    # qa_chain = RetrievalQA.from_chain_type(llm=chat_model,
    #                                     retriever=vectorDB2.as_retriever(search_type="similarity", search_kwargs={"k":2}),
    #                                     return_source_documents=True)
    
    #   qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=os.getenv("OPEN_API_KEY"), model_name='gpt-3.5-turbo'),
    #                                     retriever=vectorDB2.as_retriever(search_type="similarity", search_kwargs={"k":2}),
    #                                     return_source_documents=True)

    result = qa_chain({"query": question})
    print("???: \n")
    for page_content in (doc.page_content for doc in result["source_documents"]):
        print("???: " +page_content)
    # isResult =  result["result"] + " ".join("\n\n 참조부분: " + doc.page_content for doc in result["source_documents"])
    # return isResult
    return generate_html(result)


def generate_html(result):

    html = "<div>" + result["result"] + "</div>"
    html += "<div class='popup-container'>"
    for i, doc in enumerate(result["source_documents"]):
        html += f"""
        <div class="popup" onmousedown="document.getElementById('popuptext{i}').style.display = 'block';" onmouseup="document.getElementById('popuptext{i}').style.display = 'none';">
            참조 {i + 1}
            <span class="popuptext" id="popuptext{i}">{doc.page_content}</span>
        </div>
        """
    html += "</div>"

    return html

# def getNutrientInfo(input):
#     # persist_directory_path2 = Path(PERSIST_DIRECTORY2).resolve()
#     # print("PERSIST_DIRECTORY2(vectorDB2)의 절대 경로:", persist_directory_path2)

#     # embeddings2 = OpenAIEmbeddings(openai_api_key=os.getenv("OPEN_API_KEY"))
#     # vectorDB2 = Chroma(persist_directory=PERSIST_DIRECTORY2, embedding_function=embeddings2)
#     # print(vectorDB2)
#     # chat_history = [] #여기에 체팅 히스도리주면되려나 
#     # chain = ConversationalRetrievalChain.from_llm(llm=chat_model, retriever=vectorDB2.as_retriever(), return_source_documents=True, reduce_k_below_max_tokens=True)
#     # result = chain({"question":input, "chat_history": chat_history})
    
#     # print("is result : " + result['answer'])
#     # return result['answer']
#     persist_directory_path2 = Path(PERSIST_DIRECTORY2).resolve()
#     print("PERSIST_DIRECTORY(vectorDB2)의 절대 경로:", persist_directory_path2)

#     embeddings2 = OpenAIEmbeddings(openai_api_key=os.getenv("OPEN_API_KEY"))
#     vectorDB2 = Chroma(persist_directory=PERSIST_DIRECTORY2, embedding_function=embeddings2)
#     print(vectorDB2)
    
#     system_template="""You are an AI assistant for answering questions about nutrient.
#     You are given the following extracted parts of a long document and a question. Provide a conversational answer.
#     If you don't know the answer, Don't try to make up an answer.
#     You must have to answer in Korean.
#     You must have to cite the source
#     But Do not attribute unrelated content to the source.
#     {summaries}
#     """
#     messages = [
#         SystemMessagePromptTemplate.from_template(system_template),
#         HumanMessagePromptTemplate.from_template("{question}")
#     ]
#     prompt = ChatPromptTemplate.from_messages(messages)
#     chain_type_kwargs = {"prompt": prompt}
#     vectordbkwargs = {"search_distance": 0}
#     bk_chain = RetrievalQAWithSourcesChain.from_chain_type(
#         llm=chat_model, 
#         chain_type="stuff", 
#         retriever=vectorDB2.as_retriever(),
#         chain_type_kwargs=chain_type_kwargs,
#         reduce_k_below_max_tokens=True
#     )

#     result = bk_chain({"question": input, "vectordbkwargs": vectordbkwargs})
#     return result['answer']

def getHealthCoachingInfo(input):
    template ="""
            {reviews}에 대해서 영양코칭 AI처럼 대답해라. You are an AI assistant for answering questions about nutrient. Your name is '현마카세'
            """
    prompt = PromptTemplate(
        input_variables=["reviews"],
        template=template,
    )
    chain = LLMChain(llm=chat_model, prompt=prompt)
    return chain.run(input)

tools = [
    Tool.from_function(
        name="영양학정보",
        func=getNutrientInfo,
        description="Use it in a normal conversation, 구체적인 정보가 없을 경우 다음 tool로 넘어가라",
        return_direct=True,
    ),
    Tool.from_function(
        name="영양코칭",
        func=getHealthCoachingInfo,
        description="Use it If you don't have the appropriate tool like greating, use it last",
        return_direct=True,
    ),
    Tool.from_function(
        name="상품추천",
        func=getHealthRecoFoodSql,
        description="use it to plan or recommend food, diet, or meals. Use this tool first for diet and product recommendations",
        args_schema=RecoFoodInfo,
        return_direct=True,
    ),
]

template = '''\
    you must have to answer in korean.\
    '''

prompt = PromptTemplate(
    input_variables=[],
    template=template,
)

# Construct the react agent type.
health_agent = initialize_agent(
    # tools,
    tools=tools,
    prompt=prompt,
    llm=chat_model,
    agent="zero-shot-react-description",
    verbose=True,
    # stop=['\nObservation:'],
)


# # new action Start
# prefix = """Have a conversation with a human, You must have to answer in Korean. You have access to the following tools:"""
# suffix = """Begin!"

# {chat_history}
# Question: {input}
# {agent_scratchpad}"""

# prompt = ZeroShotAgent.create_prompt(
#     tools,
#     prefix=prefix,
#     suffix=suffix,
#     input_variables=["input", "chat_history", "agent_scratchpad"],
# )


# # llm_chain = LLMChain(chat_model, prompt=prompt)

# # run the agent
# message_history = RedisChatMessageHistory(
#     url="redis://localhost:6379/0", ttl=600, session_id="my-session"
# )

# memory = ConversationBufferMemory(
#     memory_key="chat_history", chat_memory=message_history, k=2
# )

# llm_chain = LLMChain(llm=chat_model, prompt=prompt)
# agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
# agent_chain = AgentExecutor.from_agent_and_tools(
#     agent=agent, tools=tools, verbose=True, memory=memory
# )


# user's already ate this food.
#         Compliment me for eating well and respond very simply to the advantages of eating this food{input}.
#         you only answer in korean.

def getGoodFb(input):
    template ="""
        user's already ate this food. 
        Compliment me for eating well and respond to the nutritional benefit of eating this food {input}.
        Answer in a short and simple way within 150 characters like a nutrition counselor.
        you only answer in korean.
        """
    prompt = PromptTemplate(
        input_variables=["input"],
        template=template,
    )
    chain = LLMChain(llm=chat_FeedBack_model, prompt=prompt)
    return chain.run(input)

def getbadFb(input):
    template ="""
        express your regret that user didn't eat this food, and at the end, recommend that you try it next time because of the nutritional benefits of this food {input} 
        Answer in a short and simple way within 200 characters like a nutrition counselor.
        you only answer in korean.
        """
    prompt = PromptTemplate(
        input_variables=["input"],
        template=template,
    )
    chain = LLMChain(llm=chat_FeedBack_model, prompt=prompt)
    return chain.run(input)

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
        
        
        
# Define the lists
allergyList = [
    { "name": "없음", "value": "NONE" },
    { "name": "우유 알러지", "value": "milk_allergy" },
    { "name": "계란 알러지", "value": "egg_allergy" },
    { "name": "땅콩 알러지", "value": "peanut_allergy" },
    { "name": "조개 알러지", "value": "shellfish_allergy" },
    { "name": "밀가루 알러지", "value": "gluten_allergy" },
    { "name": "새우 알러지", "value": "shrimp_allergy" },
    { "name": "복숭아 알러지", "value": "peach_allergy" },
    { "name": "토마토 알러지", "value": "tomato_allergy" },
]

diseaseList = [
    { "name": "없음", "value": "NONE" },
    { "name": "고혈압", "value": "good_for_hypertension" },
    { "name": "당뇨병", "value": "good_for_diabetes" },
    { "name": "심장질환", "value": "good_for_heart_disease" },
    { "name": "뇌졸중", "value": "good_for_stroke" },
    { "name": "암", "value": "good_for_cancer" },
    { "name": "간질환", "value": "good_for_liver_disease" },
    { "name": "폐질환", "value": "good_for_lung_disease" },
    { "name": "비만", "value": "good_for_obesity_disease" },
]

# Swap the name and value
swapped_allergy_list = {item['value']: item['name'] for item in allergyList}
swapped_disease_list = {item['value']: item['name'] for item in diseaseList}

swapped_allergy_list, swapped_disease_list

def find_name(value, dictionary):
    return dictionary.get(value, "NONE")