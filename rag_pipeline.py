from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from config import CHAT_MODEL

PROMPT_TEMPLATE = """You are a helpful assistant. Use the following context to answer the question.
If you don't know the answer from the context, say "I don't have that information in the provided documents."

Context:
{context}

Question: {question}

Answer:"""


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(vectorstore: Chroma):
    """Chain input: the question string. Output: {"question", "source_documents", "answer"}."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.3)
    prompt = PromptTemplate.from_template(PROMPT_TEMPLATE)

    answer_chain = (
        RunnablePassthrough.assign(context=lambda x: _format_docs(x["source_documents"]))
        | prompt
        | llm
        | StrOutputParser()
    )

    # Retrieve once, then feed the same documents to both the answer and the citations
    return RunnableParallel(
        source_documents=retriever,
        question=RunnablePassthrough(),
    ) | RunnablePassthrough.assign(answer=answer_chain)
