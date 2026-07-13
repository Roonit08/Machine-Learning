
# Importing the required libraries 
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
import os

#loading the environment variables form .env file
load_dotenv()
API_KEY = os.getenv("Assignment_work_chatbot_API_key")
client = Groq(api_key=API_KEY)

# page title and description
st.markdown("<h1 style='text-align: center;'>ROO</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Hello! I'm ROO, your friendly AI chatbot. How can I assist you today?</p>", unsafe_allow_html=True)

# system prompt for the chatbot 
system_prompt = {
    "role": "system",
    "content": "you're very helpful and friendly assistant. your name is ROO. "
    "Whatever user asks you, you will answer in a very friendly and helpful manner. you will answer in a very simple and easy words."
}

# Initializing the chat history in the session state if it does not exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [system_prompt]

# Displaying the chat messages from the session state 
for message in st.session_state.chat_history:
    if message["role"]!="system":
        st.chat_message(message["role"]).write(message["content"])

# taking the user input here
user_input = st.chat_input("Type your message here...")

# if the user has entered a message, send it to the chatbot and get the response
if user_input:
    # shows the user's message
    st.chat_message("user").write(user_input)
    # adding user message to the chat history
    user_query = {
        "role": "user",
        "content": user_input
    }
    st.session_state.chat_history.append(user_query)

# sending the user input to the chatbot and getting the response 
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=st.session_state.chat_history,
        max_tokens=1500,
        temperature=0.7
    )
    result = response.choices[0].message.content
 
    # displays the chatbot response
    st.chat_message("assistant").write(result)
 
    # adding the chatbot response to the chat history 
    ai_response_context = {
        "role": "assistant",
        "content": result
    }
    st.session_state.chat_history.append(ai_response_context)
