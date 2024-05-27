import streamlit as st
import os
import requests
from io import BytesIO
from PIL import Image
from typing import List, Dict

from src.database.weaviate_interface_v4 import WeaviateIndexer, WeaviateWCS
from weaviate.classes.query import Filter

from transformers import AutoModel, AutoProcessor, AutoTokenizer
from src.llm.llm_interface import LLM

# Initialize the client and LLM instances
api_key = ''
url = ''

client = WeaviateWCS(endpoint=url, api_key=api_key)
collection_name = 'FashionCLIP_sample1MM'

model_name = "patrickjohncyh/fashion-clip"
processor = AutoProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

llm = LLM(model_name="gpt-4o")

def get_text_embedding(text):
    inputs = tokenizer([text], padding=True, return_tensors="pt")
    embedding = model.get_text_features(**inputs).detach().cpu().numpy()    
    return embedding

def generate_search_tokens(user_query: str) -> str:
    system_message = """
    You are a fashion assistant. Convert the user's fashion-related query into a brief, descriptive phrase that can be used to find relevant fashion items. Use the examples below as a guide.

    Examples:
    User: "I'm looking for a black dress for a party with a 70s theme"
    Assistant: "black dress 70s party theme"

    User: "I'm travelling to Italy in summer and I'm looking for some clothes to go to the beach"
    Assistant: "beach in Italy summer clothes"

    User: "I'm going to a halloween party and want to go dressed as Kramer from Seinfeld"
    Assistant: "Outfit from Kramer from Seinfeld"

    User: "Do you have black nike shoes with a white pipe?"
    Assistant: "Black nike shoes with white pipe"
    """

    response = llm.chat_completion(system_message=system_message, user_message=user_query)
    return response

def create_fashion_prompt(conversation_history: List[Dict[str, str]], results: List[Dict[str, any]]) -> str:
    conversation = "\n".join([f"{msg['role']}: {msg['message']}" for msg in conversation_history])
    sorted_results = sorted(results, key=lambda x: x['distance'])
    top_results = sorted_results[:5]
    formatted_results = [
        f"Designer: {result['dESIGNER_SLUG']}\nShort description: {result['sHORT_DESCRIPTION']}\nColor: {result['cOLOR']}\nCategory: {result['cATEGORY']}\nImage URL: {result['iMAGE_URL']}\n"
        for result in top_results
    ]
    formatted_results_str = "\n".join(formatted_results)
    prompt = f"""
    Conversation so far:
    {conversation}
    
    Assistant: Here are the top 5 fashion items based on your latest query:
    {formatted_results_str}
    """
    return prompt

def respond_to_user_query(conversation_history: List[Dict[str, str]], results: List[Dict[str, any]]) -> str:
    fashion_prompt = create_fashion_prompt(conversation_history, results)
    system_message = """
    You are a fashion expert. Based on the context provided, help the user with fashion recommendations. 
    If there's previous conversation history, continue from where the last response left off. 
    Make the recommendations sound conversational and friendly. Mention the designer, category, color, and include a brief description. 
    Do not use any additional information about the items apart from the one provided as a context.
    In the line after the description for each item you should put the image url in parenthesis, like this:
    "First up, you can't go wrong with a pair of classic black Converse sneakers. They're comfy, stylish, and perfect for jumping around in the mosh pit.\n
    (http://lyst-static.s3-eu-west-1.amazonaws.com/photos/dsw/855fad42/david-tate-Bone-Madelyn-Wedge-Sandal.jpeg)"
    Ensure the response is informal and doesn't include bullet points or titles. Here's the context:
    """
    response = llm.chat_completion(system_message=system_message, user_message=fashion_prompt)
    return response

def display_response_with_images(response: str):
    import re
    def replace_urls_with_images(match):
        url = match.group(1)
        return f'<img src="{url}" width="200">'
    response_with_images = re.sub(r'\((http[^)]+)\)', replace_urls_with_images, response)
    st.markdown(f"<b>Fashion Bot:</b> {response_with_images}", unsafe_allow_html=True)

# Streamlit app layout
st.title("Fashion Assistant")
st.write("Ask any fashion-related questions and get recommendations!")

genders = ['M', 'F']

selected_gender = st.selectbox("Select Gender", genders)

filter = Filter.by_property('gENDER').equal(selected_gender)

if 'conversation' not in st.session_state:
    st.session_state.conversation = []

def add_message_to_conversation(role, message):
    st.session_state.conversation.append({"role": role, "message": message})

# Display chat history
for chat in st.session_state.conversation:
    if chat['role'] == 'user':
        with st.container():
            st.markdown(f"<div style='text-align: right; background-color: #f0f0f0; padding: 10px; border-radius: 5px;'><b>You:</b> {chat['message']}</div>", unsafe_allow_html=True)
    else:
        with st.container():
            display_response_with_images(chat['message'])

if 'user_query' not in st.session_state:
    st.session_state.user_query = ''

def submit():
    st.session_state.user_query = st.session_state.input
    st.session_state.input = ''

input_container = st.empty()
with input_container:
    user_query = st.text_input("", placeholder="Type your message here...", key="input", label_visibility="collapsed", on_change=submit)

if st.session_state.user_query:
    add_message_to_conversation('user', st.session_state.user_query)
    
    # Generate search tokens
    search_tokens = generate_search_tokens(st.session_state.user_query)
    
    # Get text embedding and query the vector search API
    text_embedding = get_text_embedding(search_tokens)[0].tolist()
    display_properties = [prop.name for prop in client.show_collection_properties(collection_name)]
    vector_response = client.vector_search(
        request=None,
        query_vector=text_embedding,
        collection_name=collection_name,
        limit=5, 
        return_properties=display_properties,
        return_raw=False,
        device='cpu',
        filter=filter
    )
    
    print(st.session_state.conversation)
    # Get the response from OpenAI API
    response = respond_to_user_query(st.session_state.conversation, vector_response)
    
    # Add assistant's response to the conversation
    add_message_to_conversation('assistant', response)
    
    # Clear the user query
    st.session_state.user_query = ''
    st.rerun()

# Align the text input and send button to the right using CSS
st.markdown("""
    <style>
    div[data-testid="stTextInput"] > div > div {
        display: flex;
        justify-content: flex-end;
    }
    div[data-testid="stTextInput"] > div > div > input {
        text-align: right;
    }
    div[data-testid="stButton"] > div {
        display: flex;
        justify-content: flex-end;
    }
    </style>
    """, unsafe_allow_html=True)
