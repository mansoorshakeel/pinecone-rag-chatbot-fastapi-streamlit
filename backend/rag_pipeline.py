import os
from typing import Dict, List

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index_name = os.environ["PINECONE_INDEX"]
    return PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)


def load_chat_model():
    repo_id = os.getenv("HF_LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")

    assert token, "HF token not found. Set HUGGINGFACEHUB_API_TOKEN in .env"

    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="text-generation",
        huggingfacehub_api_token=token,   # ✅ force token use
        max_new_tokens=512,
        do_sample=False,
    )
    return ChatHuggingFace(llm=llm)


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[source={d.metadata.get('source','')}, page={d.metadata.get('page','')}] \n{d.page_content}"
        for d in docs
    )

def format_history(history) -> str:
    if not history:
        return ""
    # keep it small: only last 6 messages (3 user+assistant turns)
    history = history[-6:]
    lines = []
    for m in history:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "user")
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines)

def answer_question(question: str, history=None) -> Dict:
    vs = load_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 4})
    docs = retriever.invoke(question)

    context = format_docs(docs)
    history_text = format_history(history)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful RAG assistant. Use ONLY the provided context to answer.\n"
         "If the answer is not in the context, say: 'I don't know based on the documents.'\n"
         "Chat history is for continuity, not as a factual source unless supported by context."),
        ("user",
         "Chat history:\n{history}\n\n"
         "Question: {question}\n\n"
         "Context:\n{context}")
    ])

    model = load_chat_model()
    chain = prompt | model | StrOutputParser()

    answer = chain.invoke({"question": question, "context": context, "history": history_text})

    return {"answer": answer, "sources": [d.metadata for d in docs]}

