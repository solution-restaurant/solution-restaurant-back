import re
import os
from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore.document import Document
import os
print(os.getcwd())
FILE_NAME = "../pdf/greating_data_korean_new.csv"
PERSIST_DIRECTORY = 'db'

loader = CSVLoader(FILE_NAME)
documents = loader.load()
output = []
# text 정제
for page in documents:
    text = page.page_content
    text = re.sub('\n', ' ', text)  # Replace newline characters with a space
    text = re.sub('\t', ' ', text)  # Replace tab characters with a space
    text = re.sub(' +', ' ', text)  # Reduce multiple spaces to single
    output.append(text)
print(output)
doc_chunks = []
for line in output:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, # 최대 청크 길이
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""], #  텍스트를 청크로 분할하는 데 사용되는 문자 목록
        chunk_overlap=0, # 인접한 청크 간에 중복되는 문자 수
    )
    chunks = text_splitter.split_text(line)
    for i, chunk in enumerate(chunks):
        doc = Document(
            page_content=chunk, metadata={"page": i, "source": FILE_NAME}
        )
        doc_chunks.append(doc)
        
    
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPEN_API_KEY"))
index = Chroma.from_documents(documents=doc_chunks, embedding=embeddings, persist_directory=PERSIST_DIRECTORY)