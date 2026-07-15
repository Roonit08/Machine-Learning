

from dotenv import load_dotenv
import os
from pinecone import Pinecone
from create_vectors import embed_text,vector_index
from groq import Groq
import streamlit as st

load_dotenv()

groq_client = Groq(api_key=os.environ.get("Assignment_work_chatbot_API_key"))

st.title("ROO - AI Chatbot with RAG")
st.subheader("Ask questions about your documents")

user_query = st.text_input("Enter your question here:")

send_btn = st.button("Send")

system_prompt = {
    "role": "system",
    "content": "You're a helpful and friendly assistant named ROO. Answer user questions in a simple and "
    "easy-to-understand manner."}
if send_btn and user_query:
    #embed the user query
    query_vector = embed_text(user_query)

    #retrieve relevant documents from Pinecone
    response = vector_index.query(
        vector=query_vector,
        top_k=2, #Retrieve top 2 relevant documents
        include_metadata=True
    )

    # Extract the relevent documents
    similar_documents = ""
    for match in response["matches"]:
        similar_documents += match["metadata"]["text"] + "\n\n"

    #Prepare the context for Groq LLM
    user_prompt={
       "role":"user",
       "content":f"""Here ia relevent documents fetched according to the user question,
       use this document to analyze and answer user query):{similar_documents}
Question:{user_query}"""
    }

    # Generate answer using Groq LLM 
    Groq_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[system_prompt,user_prompt],
        max_tokens=1500,
        temperature=0.7
    )
    result = Groq_response.choices[0].message.content

    st.subheader("Answer:")
    st.write(result)

