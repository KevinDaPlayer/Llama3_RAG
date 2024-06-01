from flask import Flask, request, jsonify
import asyncio
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

app = Flask(__name__)

# 设置 PDF 文件路径
PDF_FILE_PATH = r'C:\Users\32588\PycharmProjects\Llama_RAG\statics\pdfs\icd10cm_tabular_2023_test.pdf'

# 初始化 LangChain 组件
loader = PyPDFLoader(PDF_FILE_PATH)
docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
retriever = vectorstore.as_retriever()

llm = ChatOllama(model="llama3", format="json" , temperature=0.3)
prompt = PromptTemplate(
    template="""system You are an experienced physician and medical coder with extensive knowledge of the ICD-10 classification system. Your task is to predict the most relevant top five ICD-10 codes based on the provided case description, considering symptoms, medical history, and other pertinent information. Ensure your answer is both accurate and educational, enhancing understanding of the ICD-10 system. Please list the top five potential ICD-10 codes and briefly explain the disease or condition each code corresponds to, 請依照此格式範例生成，以下為範例: K35.8 :疾病解釋;I10: 疾病解釋;E11.9 :疾病解釋;J45.909 :疾病解釋;M54.2 :疾病解釋。檔案名稱為'topFiveICD10Codes', 內容請包含icd-10 cm code('code')、疾病簡介('description')、疾病詳細解釋('detail'), please answer with Json. user Question: {question} Context: {context} Answer: assistant""",
    input_variables=["question", "context"],
)

@app.route('/rag', methods=['POST'])
def handle_request():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': 'No question provided'}), 400

    answer = asyncio.run(rag_process(question))
    print(answer)
    return jsonify(answer)

async def rag_process(question):
    retrieved_docs = retriever.invoke(question)
    context = " ".join([doc.page_content for doc in retrieved_docs])
    full_prompt = prompt.format(question=question, context=context)
    response_stream = llm.astream(full_prompt)
    full_response = ""
    async for resp_part in response_stream:
        full_response += resp_part.content

    parsed_response = JsonOutputParser().parse(full_response)
    return parsed_response

if __name__ == '__main__':
    app.run(debug=True)
