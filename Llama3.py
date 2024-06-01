from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
#from langchain.callbacks import CallbackManager, BaseCallbackManager
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import ollama
import asyncio
from flask import Flask

# app = Flask(__name__)


# 這裡設定你的 PDF 文件的本地路徑
PDF_FILE_PATH = r'C:\Users\32588\PycharmProjects\Llama_RAG\statics\pdfs\icd10cm_tabular_2023_test.pdf'

loader = PyPDFLoader(PDF_FILE_PATH)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print(splits)
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()
print("Vector store and retriever initialized.")

llm = ChatOllama(model="llama3", format="json")

prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an experienced physician and medical coder with extensive knowledge of the ICD-10 classification system. Your task is to predict the most relevant top five ICD-10 codes based on the provided case description, considering symptoms, medical history, and other pertinent information. Ensure your answer is both accurate and educational, enhancing understanding of the ICD-10 system. Please list the top five potential ICD-10 codes and briefly explain the disease or condition each code corresponds to, please answer with Json . <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context"],
    )


async def rag_process(question):
    retrieved_docs = retriever.invoke(question)
    print("Retrieved docs:", retrieved_docs)
    print("Sample document object:", retrieved_docs[0].__dict__)

    context = " ".join([doc.page_content for doc in retrieved_docs])
    full_prompt = prompt.format(question=question, context=context)
    response_stream = llm.astream(full_prompt)
    full_response = ""
    async for resp_part in response_stream:
        full_response += resp_part.content

    print(full_response)
    parsed_response = JsonOutputParser().parse(full_response)
    return parsed_response

question = "This 65-year-old male has type 2 diabetes mellitus with diabetic chronic kidney disease stage 3 with insulindependent. He suffered from right chest pain after traffic accident on 7/20. According to the patient, he wasriding scooter collides with car. He came to our emergency department for help. Then right 4-7th ribsfracture, right hemopneumothorax and laceration of right lower leg were impression, during hospitalizationon right chest tube and sutured skin of laceration site were performed."

answer = asyncio.run(rag_process(question))
print(answer)

# Define API endpoint for ICD-10 prediction

#
# # Start the Flask application
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)
