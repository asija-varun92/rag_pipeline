from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()


class RagSearch():
    def __init__(self, retriever):
        # Initialize the ChatGroq model with the API key from environment variables
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1, max_tokens=1024)
        self.rag_retriever = retriever


    def search(self, query, top_k = 3):
        ## retrieve the context
        results = self.rag_retriever.retrieve(query, top_k=top_k)

        context = "\n\n".join([doc["content"] for doc in results]) if results else ""
        sources = [{"source": doc["metadata"].get("source_file"),
                    "page":  doc["metadata"].get("page")
                    }
                    for doc in results]
        if not context:
            return "No relevant context found to answer the question"
        ## generate the answer using GROQ LLM
        prompt = f"""
        You are a question-answering assistant.

        Answer ONLY using the information provided in the context below.

        If the answer cannot be found in the context, reply exactly:

        "I don't have enough information in the provided context."

        Do not use your own knowledge.

        Context:
        {context}

        Question:
        {query}

        Answer:
        """
        response=self.llm.invoke([prompt.format(context=context, query=query)])
        output = {"answer": response.content,
                "sources": sources}
        return output
