# Import Statements
import os
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
from result_handler import handle_file_upload
from embeddings import ensemble_retriever

# Configuration
load_dotenv()
USER_NAME = "user"
ASSISTANT_NAME = "assistant"
ASSISTANT_AVATAR = './tb.png'
model = "azure_openai_app"

# Initialize the Azure OpenAI service
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_API_KEY")
client = AzureOpenAI(api_version="2023-03-15-preview")

# Function to get response from LLM model (Chat-GPT)
def response_chatgpt(user_msg: str, input_documents, chat_history: list = []):
    system_msg = (
        # """あなたは世界中で信頼されているQAシステム「KANTくん」です。
        # トヨタ紡織の標準に関するクエリに対して最高のアドバイスを1つにまとめる

        # 従うべきルール:
        # ・コンテキスト以外の情報から回答を生成してはいけない
        # ・参照したファイル名、標準名、ページなどの情報は絶対に回答に含めてはいけない
        # """
        """"You are an Assistant named TB-RAG. Answer the questions in detail based on the provided document. "
        "If the information is not in the documents or you can't find it, say to user you don't know the answer."
        "at the end of response cite the resource in standard way. Don't hallucinate"""
    )
    messages = [{"role": "system", "content": system_msg}]

    for chat in chat_history:
        messages.append({"role": chat["name"], "content": chat["msg"]})

    messages.append({"role": USER_NAME, "content": user_msg})

    for doc in input_documents:
        messages.append({"role": "user", "content": f"Document snippet:\n{doc['content']}"})

    try:
        response = client.chat.completions.create(model=model, messages=messages, temperature=0)
        return {
            "answer": response.choices[0].message.content,
            "sources": input_documents
        }
    except Exception as e:
        st.error(f"Could not find LLM model: {str(e)}")
        return None



def main():
    st.title("TB Assist")
    st.write("Please upload your file and type a message")
    
    # File Upload Sidebar option
    with st.sidebar:
        st.title('Document Chat Loader')
        files = st.file_uploader("Upload your file", accept_multiple_files=True, type=['pdf', 'txt', 'docx', 
                                                                                       'pptx', 'xlsx', 'csv'])
        send_button = st.button("Submit", key="send_button")
        if send_button:
            # try:
            text_chunks = handle_file_upload(files)
            file_names = [file.name for file in files]
            if text_chunks:
                st.session_state.file_name = file_names
                st.session_state.text_chunks = text_chunks
            # except Exception as e:
            #     st.error(f"Please upload a valid pdf file")

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    user_msg = st.chat_input("Enter your message here", key="user_input")
    if user_msg:
        for chat in st.session_state.chat_log:
            with st.chat_message(chat["name"], avatar=ASSISTANT_AVATAR if chat["name"] == ASSISTANT_NAME else None):
                st.write(chat["msg"])
                if "citations" in chat:
                    for citation in chat["citations"]:
                        st.write(f"- {citation}")

        with st.chat_message(USER_NAME):
            st.write(user_msg)
        reranked_results=ensemble_retriever(st.session_state.text_chunks, user_msg)
        doc_texts = [{"content": doc.page_content, "metadata": doc.metadata} for doc in reranked_results]
        with st.spinner("Loading answer..."):
            response = response_chatgpt(user_msg, doc_texts, chat_history=st.session_state.chat_log)
            if response:
                with st.chat_message(ASSISTANT_NAME, avatar=ASSISTANT_AVATAR):
                    assistant_msg = response["answer"]
                    assistant_response_area = st.empty()
                    assistant_response_area.write(assistant_msg)

                    # citations = []
                    # info_missing = ["I don't know", "I couldn't find", "there is no information about", 
                    #                 "I'm sorry", "is not mentioned", "I don't have information", 
                    #                 "there is no specific information", "there is no mention", 
                    #                 "document does not provide"]
                    # if not any(response in assistant_msg for response in info_missing):
                    #     st.write("## Citations")
                    #     if response["sources"]:
                    #         for idx, source in enumerate(response["sources"], start=1):
                    #             metadata = source["metadata"]
                    #             file_name = st.session_state.file_name if "file_name" in st.session_state else "Source unavailable"
                    #             content = source["content"]
                                
                    #             if 'page_number' in metadata:
                    #                 page_number = metadata['page_number']
                    #                 for file_nam in file_name:
                    #                     if file_nam.endswith(('.pdf', '.pptx', '.ppt', '.doc', '.docx')):
                    #                         citations.append(f"File: {file_nam}---page: {page_number}")
                    #                     elif file_nam.endswith(('.xlsx', '.xls', 'csv')):
                    #                         citations.append(f"File: {file_nam}---row: {page_number}")

                    st.session_state.chat_log.append({"name": USER_NAME, "msg": user_msg})
                    # st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": assistant_msg, "citations": citations})
                    st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": assistant_msg})
                    # # Display citations for the current response
                    # for citation in citations:
                    #     st.write(f"- {citation}")

if __name__ == "__main__":
    main()