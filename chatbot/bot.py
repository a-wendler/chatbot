import streamlit as st
from llama_index import VectorStoreIndex, ServiceContext
from llama_index.llms import Perplexity
from llama_index import SimpleDirectoryReader

@st.cache_resource(show_spinner=True)
def load_data():
    with st.spinner(text="Die LSB-Informationen werden indiziert. Das dauert nur ein paar Augenblicke."):
        reader = SimpleDirectoryReader(input_dir="chatbot/data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=llm)
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

pplx_api_key = st.secrets.pplx_key
llm = Perplexity(
    api_key=pplx_api_key, model="pplx-70b-chat", temperature=0.4, system_prompt="Du bist ein Experte für die Leipziger Städtischen Bibliotheken. Du hilfst Nutzerinnen und Nutzern dabei, die Bibliothek zu benutzen. Du beantwortest Fragen zum Ausleihbetrieb, zu den Standorten und den verfügbaren Services. Deine Antworten sollen auf Fakten basieren. Halluziniere keine Informationen über die Bibliotheken, die nicht auf Fakten basieren. Wenn Du eine Information über die Bibliotheken nicht hast, sage den Nutzenden, dass Du Ihnen nicht weiterhelfen kannst. Antworte auf Deutsch."
)

st.header("Der LSB-Service-Chat 💬 📚")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Was möchten Sie über die Leipziger Städtischen Bibliotheken wissen?"}
    ]



index = load_data()
chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

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