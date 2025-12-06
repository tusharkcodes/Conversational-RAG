from langchain_core.prompts import ChatPromptTemplate

prompt =  ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant that provides concise and accurate answers based self help context."),
        ("user", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    ]
)

result = prompt.invoke({"context": "Self help", "question": "What is self help?"})

print(result)