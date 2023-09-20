import fa_config
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import qdrant_client

q_url = fa_config.qdrant_host

client = qdrant_client.QdrantClient(
    url=q_url,
    api_key=fa_config.qdrant_key,
)

#create a vector store
def get_vector_store():
    embeddings = OpenAIEmbeddings(openai_api_key=fa_config.openai_key)
    vectorstore = Qdrant(
        client=client,
        embeddings=embeddings, 
        collection_name='dpo_chat_bot')
    return vectorstore

#get conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(openai_api_key=fa_config.openai_key)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain

def get_chain():
    vectorstore = get_vector_store()
    print('I got the vector too')
    conversation_chain = get_conversation_chain(vectorstore)
    print('Conversation chain too is set')
    return conversation_chain



