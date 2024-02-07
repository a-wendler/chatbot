import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

openai.api_key = st.secrets.openai_key
st.header("Der Loga-Service-Chat 💬 ⌛ ⌚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Was möchten Sie über das Personalportal Loga wissen?"}
    ]

@st.cache_resource(show_spinner=True)
def load_data():
    with st.spinner(text="Die Loga-Informationen werden indiziert. Das dauert nur ein paar Augenblicke."):
        reader = SimpleDirectoryReader(input_dir="loga-data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="Du bist Kundenberater für das Personalportal Loga. In dem Personalportal können Mitarbeiterinnen und Mitarbeiter der Stadt Leipzig ihre Arbeitszeit erfassen, Anträge auf Urlaub und Zeitausgleich stellen. Auch Krankmeldungen werden im Personalportal Log erfasst. Deine Antworten sollen auf Fakten basieren. Halluziniere keine Informationen über das Personalportal, die nicht auf Fakten basieren. Wenn Du eine Information über das Personalportal nicht hast, sage den Nutzenden, dass Du Ihnen nicht weiterhelfen kannst. Antworte immer auf Deutsch."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()
# chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
chat_engine = index.as_chat_engine(chat_mode="context", verbose=True)

if prompt := st.chat_input("Ihre Frage"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Ich denke nach ..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
