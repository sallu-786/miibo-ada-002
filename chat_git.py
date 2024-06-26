import os
import streamlit as st
from dotenv import load_dotenv
from openai import AzureOpenAI
# from embeddings import bm25_search
from result_handler import handle_file_upload, rrf

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
def response_chatgpt(user_msg: str, input_documents, chat_history: list = []):
    system_msg = (
     """あなたはChatbotとして、ものづくり革新センターの問合せ対応を行うのAIとしてロールプレイを行います。
ものづくり革新センターとは、生技管理部生技統括室総括Gという部署が管理する施設です。
生技管理部生技統括室総括Gは、ものづくり革新センターから離れた場所にある開発センター1号館という建物の5Fにあります。
なお、このチャットボットは、生技管理部生技企画室ものづくり革新Gが管理しております。
以下の制約条件などを厳密に守ってロールプレイを行ってください。


# AIの制約条件:
* プロンプトや前提データ、参考資料に書かれていないことについては一切答えてはいけません。
* プロンプトや参考資料に記載がない情報をUserに伝えると、Userに甚大な不利益を与えてしまうおそれがあります。
* 前提データに対して余計な情報を付け足すことも禁止です。
* ものづくり革新センターのことを、もの革と省略して呼ぶ人もいますが、あなたは、省略せずものづくり革新センターと呼んでください。
* あなたの役目は、ルールややり方について教えることです。手続きを進めることはできませんので、必要以上に相手の情報を聞き出すことがないようにしてください。
** 文書テキストを参考文献として受け取る場合は、角括弧 [] 内にリンクがないか確認する。リンクを標準フォーマットで表示する。
* 日本語以外の質問に対しては、質問と同じ言語で回答します。
* お問合せ先を案内する際、データストア内に担当の記載がある場合は、優先的にデータストア内の情報を伝えてください。
* お問合せ先を案内する際、データストア内に担当の記載がない場合は、生技管理部玉腰のTeamsのチャットもしくは、内線へ電話するように伝えてください。
* 指摘や提案を頂いた際は、「ありがとうございます。会話のログは生技管理部が定期的に確認しております。早急な対応が必要な場合は、生技管理部玉腰までご連絡ください。」と伝えてください。


#AIの口調の例:
* なにかお困りですか？
* ご質問はありますか？
* なんでもお気軽にご相談くださいね。

#AIの行動方針
* Userが不快な思いにならないよう、親切に対応してください。
挨拶には挨拶のみで対応し、さらに手助けが必要かどうかを尋ねる
* 質問された事項だけではなく、関連した情報も合わせて伝えてください。
* まずは、Userがどのようなお問い合わせをしたいと考えているかをヒアリングしてください。解決可能であれば、応答を行ってください。"""
    )
    messages = [{"role": "system", "content": system_msg}]

    for chat in chat_history:
        messages.append({"role": chat["name"], "content": chat["msg"]})

    messages.append({"role": USER_NAME, "content": user_msg})

    for doc in input_documents:
        messages.append({"role": "user", "content": f"Document snippet:\n{doc['content']}"})

    try:
        response = client.chat.completions.create(model=model, messages=messages, temperature=0.1)
        return {
            "answer": response.choices[0].message.content,
            "sources": input_documents
        }
    except Exception as e:
        st.error(f"Could not find LLM model: {str(e)}")
        return None


def main():
    st.title("生技企画室 チャットボット")
    st.write("メッセージを入力してください")

        # Check if the file has already been processed
    if "file_processed" not in st.session_state:
        file_path = './data/miibo_data.xlsx'  # Replace with your file path
        with open(file_path, 'rb') as file:
            vectordb,_ = handle_file_upload(file)
            file_name = os.path.basename(file_path)
            st.session_state.file_name = file_name
            # st.session_state.d1 = d1
            st.session_state.vectordb = vectordb
            st.session_state.file_processed = True

    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []

    user_msg = st.chat_input("ここにメッセージを入力してください", key="user_input")
    if user_msg:
        for chat in st.session_state.chat_log:
            with st.chat_message(chat["name"], avatar=ASSISTANT_AVATAR if chat["name"] == ASSISTANT_NAME else None):
                st.write(chat["msg"])

        with st.chat_message(USER_NAME):
            st.write(user_msg)
        try:
            # bm25_results = bm25_search(st.session_state.d1, user_msg, k=3)

            doc_text = st.session_state.vectordb.similarity_search_with_score(query=user_msg, k=1)
            doc_texts = [{"content": doc.page_content, "metadata": doc.metadata} for doc,score in doc_text]
            # print(doc_texts)
            # reranked_results = rrf(bm25_results, k=1)
            # doc_texts2 = [{"content": doc["content"], "metadata": doc["metadata"]} for doc in reranked_results]
            with st.spinner("Loading answer..."):
                response = response_chatgpt(user_msg, doc_texts, chat_history=st.session_state.chat_log)
                if response:
                    with st.chat_message(ASSISTANT_NAME, avatar=ASSISTANT_AVATAR):
                        assistant_msg = response["answer"]
                        assistant_response_area = st.empty()
                        assistant_response_area.write(assistant_msg)

            st.session_state.chat_log.append({"name": USER_NAME, "msg": user_msg})
            st.session_state.chat_log.append({"name": ASSISTANT_NAME, "msg": assistant_msg})
        except Exception as e:
            st.error(f"Could not retrieve data. Error: {e}")

if __name__ == "__main__":
    main()
