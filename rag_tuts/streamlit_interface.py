import streamlit as st
import requests, json

# Streamlit app title
st.title('Oreo Chatbot')
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# Input field for user to add a question
user_question = st.text_input('Enter your question:')

# Button to submit the question
if st.button('Submit'):
    # Make a POST request to your API
    response = requests.post('http://localhost:5000/chatbot', json={'query': user_question})
    
    # Display the response from the API
    try:
        data = response.json()
        st.write('Response:', data['response'])
        st.write('Is from cache:', data['is_from_cache'])
        st.write('Time taken:', data['time_taken'])
    except json.decoder.JSONDecodeError:
        st.write('Error: Invalid JSON response from API')
    except KeyError as e:
        st.write(f'Error: Missing key in JSON response: {e}')
    except Exception as e:
        st.write(f'Error: {e}')
