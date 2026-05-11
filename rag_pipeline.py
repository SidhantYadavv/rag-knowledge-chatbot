from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

PROMPT_TEMPLATE = """You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer from the context, say "I don't have that information in the provided documents."

Context:
{context}

Question: {question}

Answer:"""

def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_qa_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=PROMPT_TEMPLATE,
    )

    chain = RunnableParallel(
        result=(
            RunnablePassthrough.assign(
                context=lambda x: _format_docs(retriever.invoke(x["query"]))
            )
            | RunnablePassthrough.assign(question=lambda x: x["query"])
            | prompt
            | llm
            | StrOutputParser()
        ),
        source_documents=lambda x: retriever.invoke(x["query"]),
    )

    return chain
