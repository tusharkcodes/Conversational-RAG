# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS

# FAISS_DB_PATH = "faiss_selfhelp_db"
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# vectorstore = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)

# while True:
#     q = input("\nAsk about habits/self-help (or type 'quit'): ")
#     if q.lower() in ["quit", "exit"]:
#         break
#     results = vectorstore.similarity_search(q, k=5)
#     print("\n" + "="*60)
#     for i, doc in enumerate(results, 1):
#         print(f"{i}. [{doc.metadata.get('book', 'Unknown')}]")
#         print(doc.page_content.strip()[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
#         print("-" * 50)  


# rag_app.py  ← Your final RAG application (run this forever)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate , MessagePlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from model import Response

load_dotenv()  

# ==================== CONFIG ====================

FAISS_DB_PATH = "faiss_selfhelp_db"  

parser = PydanticOutputParser(pydantic_object=Response)

format_instructions = parser.get_format_instructions()


llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)


embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(FAISS_DB_PATH, embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # top 6 relevant chunks


prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are an expert self-help coach. "
     "Answer the question using ONLY the context below from books like Atomic Habits, Deep Work, etc. "
     "Keep answers concise, practical, and actionable. "
     "I want the output response in the given format"
     "If the context doesn't have the answer, say: 'I don't have information about that in my current books.'"),
     
    ("human", 
     "Context from books:\n"
     "{context}\n\n"
     "Question: {question}\n"
     "response_format:{format_instructions}\n\n"
     "Answer in a friendly and motivating tone:")
])

# ==================== RAG CHAIN ====================

def get_context(query):
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join([f"From 《{doc.metadata.get('book', 'Unknown Book')}》\n{doc.page_content}" for doc in docs])

chain = prompt | llm | parser 

question = "How can I build better habits according to Atomic Habits?"

context = get_context(question)
    
response = chain.invoke({
        "context": context,
        "question": question,
        "format_instructions": format_instructions 
    })

chat_history = []

chat_history.extend([
        HumanMessage(content = question),
        AIMessage(context = response)
    ])
    

# ==================== MAIN LOOP ====================
# print("Self-Help RAG Coach Ready! (type 'quit' to exit)\n")
# while True:
    # question = input("Ask me anything about habits, productivity, focus, etc.: ").strip()
    # if question.lower() in ["quit", "exit", "bye"]:
    #     print("Keep building those atomic habits!")
    #     break
    # if not question:
    #     continue

    # print("\n" + "="*70)
    # print("HeadLine",response.headline , "\n")
    # print("SubHeadline=",response.SubHeadline , "\n")
    # print("Point1",response.point1 , "\n")
    # print("point2",response.point2 , "\n")
    # print("point3",response.point3 , "\n")
    # print("Summary",response.summary , "\n")

    # print("="*70 + "\n")