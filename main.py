import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

load_dotenv()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

embedding = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
llm = ChatOpenAI(model='gpt-4')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_vectorstore_from_pdf(filepath):
    loader = PyPDFLoader(filepath)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, embedding=embedding)
    return vectorstore

def create_rag_chain(vstore):
    retriever = vstore.as_retriever()

    def retrieve(state):
        question = state["messages"][-1].content
        docs = retriever.invoke(question)
        return {"messages": state["messages"], "docs": docs}

    def generate(state):
        docs = state["docs"]
        question = state["messages"][-1].content
        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = f"Answer the following question using the context below:\n\nContext:\n{context}\n\nQuestion: {question}"
        response = llm.invoke(prompt)
        return {"messages": state["messages"] + [AIMessage(content=response.content)]}

    # ✅ Define the required schema
    class GraphState(TypedDict):
        messages: List[BaseMessage]
        docs: list

    builder = StateGraph(GraphState)  # ✅ Correct instantiation
    builder.add_node("retrieve", RunnableLambda(retrieve))
    builder.add_node("generate", RunnableLambda(generate))
    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile()

# Load default PDF
vectorstore = create_vectorstore_from_pdf("document.pdf")
graph = create_rag_chain(vectorstore)

@app.route('/', methods=['GET', 'POST'])
def index():
    global vectorstore, graph
    answer = None
    if request.method == 'POST':
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                vectorstore = create_vectorstore_from_pdf(filepath)
                graph = create_rag_chain(vectorstore)
                return redirect(url_for('index'))
        if 'query' in request.form:
            question = request.form['query']
            result = graph.invoke({"messages": [HumanMessage(content=question)]})
            answer = result['messages'][-1].content
    return render_template('index.html', answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
