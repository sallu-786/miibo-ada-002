import streamlit as st
from embeddings import get_file, get_text_chunks

# when file is uploaded by user, create new vector data for that file
def create_new_vector_db(file):
    with st.spinner("Creating vector data"):
        text = get_file(file)
        text_chunks = get_text_chunks(text)
    return text_chunks

def handle_file_upload(file):
    if file:
        text_chunks = create_new_vector_db(file)
        st.write("Vector data created successfully.")
        return text_chunks

    else:                             
        pass


