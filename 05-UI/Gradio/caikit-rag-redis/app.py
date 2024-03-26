inference_server_url = "https://demo-2-plus-demo.apps.openshiftai2.acic.local"
infer_url = f"{inference_server_url}/api/v1/task/text-generation"
str_infer_url = f"{inference_server_url}/api/v1/task/server-streaming-text-generation"
redis_url = "redis://default:j5kL7T3W@my-doc-headless.redis.svc.cluster.local:16735"
index_name = "dellwebdocs"
schema_name = "redis_schema_ai.yaml"
model_id = "test"



from typing import Any, Iterator, List, Mapping, Optional, Union
from warnings import warn
from caikit_nlp_client import HttpClient
from langchain.prompts import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.redis import Redis
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema.output import GenerationChunk
from langchain.chains import RetrievalQA
#from langchain.callbacks.streaming stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.streamlit.streamlit_callback_handler import StreamlitCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts.chat import SystemMessagePromptTemplate
from langchain.chains import LLMChain



class CaikitLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "llama-2-fb-chat-hf"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")


        client = HttpClient(inference_server_url, verify=False)
        return client.generate_text(
            model_id,
            prompt,
            preserve_input_text=False,
            max_new_tokens=50,
            min_new_tokens=5,
            timeout=36000.0,
        )
    
embeddings = HuggingFaceEmbeddings()
rds = Redis.from_existing_index(
    embeddings,
    redis_url=redis_url,
    index_name=index_name,
    schema=schema_name
)

template="""<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant named DellDigitalAssistant answering questions about any queries realated to Dell Technologies , specially related to Dell Validated solution for microsoft sql server .
You will be given a question you need to answer, and a context to provide you with information. You must answer the question based as much as possible on this context.
Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
<</SYS>>
History: {chat_history}
Question: {question}
Context: {context} [/INST]</s>
"""

prompt = PromptTemplate.from_template(template)

llm = CaikitLLM()

memory = ConversationSummaryBufferMemory(
    llm=llm,
    output_key='answer',
    memory_key='chat_history',
    return_messages=True)

retriever = rds.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4, "distance_treshold": 0.5, "include_metadata": True})

chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    memory=memory,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    get_chat_history=lambda h : h,
    verbose=False)
    
import gradio as gr
import random
import time

def respond(question, history):
    result = chain.invoke({"question": question, "chat_history": history})
    return result['answer']



demo = gr.ChatInterface(
    respond,  # Your chatbot function
    examples=["What is ObjectScale?", "What is OpenshiftAI?", "merhaba"],  # Sample inputs for the chatbot
    title="ACIC ROME BOT",  # Title for the interface
    theme="default",
)
if __name__ == "__main__":
    demo.queue().launch(
        server_name='0.0.0.0',
        share=False,
        favicon_path='./assets/conversation.ico'
        )
