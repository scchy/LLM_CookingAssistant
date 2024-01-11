# python3
# Create Date: 2024-01-10
# Author: Scc_hy
# Func: web demo
# ==============================================================================
import os
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import warnings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import gradio as gr
from openxlab.model import download
import openxlab
import json
from langchainPrepare.LLM import InternLM_LLM
from langchainPrepare.persistentVector import file2Chroma2local
import sys
import pysqlite3
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb

with open('./langchainPrepare/_k.json', 'r') as f:
    key_dict = json.load(f)


warnings.filterwarnings('ignore')
openxlab.login(ak=key_dict['ak'], sk=key_dict['sk'])

# 数据准备
persist_directory = file2Chroma2local()
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 模型下载
# /home/xlab-app-center/.cache/model or /home/.cache/model
lm_7b_path = "/home/xlab-app-center/InternLM-chat-7b"
sentence_tf_path = '/home/xlab-app-center/sentence-transformer'
if not os.path.exists(lm_7b_path):
    download(model_repo='OpenLMLab/InternLM-chat-7b', output=lm_7b_path)

print("os.listdir(lm_7b_path-father)=", os.listdir(lm_7b_path.rsplit('/', 1)[0]))
print("os.listdir(lm_7b_path)=", os.listdir(lm_7b_path))
print("os.listdir(/home/xlab-app-center/.cache/model)=", os.listdir('/home/xlab-app-center/.cache/model'))

def load_chain():
    # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name=sentence_tf_path)
    # 向量数据库持久化路径
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print('>>>>>>>>> [ 1- 加载向量数据库 ]( 完成 )')
    # 加载自定义 LLM
    llm = InternLM_LLM(model_path = lm_7b_path)
    print('>>>>>>>>> [ 2- 加载InternLM_LLM ]( 完成 )')
    # 定义一个 Prompt Template
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。尽量使答案简明扼要。总是在回答的最后说“谢谢你的提问！”。
    {context}
    问题: {question}
    有用的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],template=template)
    print('>>>>>>>>> [ 2- 实例化Prompt Template ]( 完成 )')

    # 运行 chain
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    print('>>>>>>>>> [ 3- 构建检索问答链 ]( 完成 )')
    return qa_chain


class Model_center():
    """
    存储检索问答链的对象 
    """
    def __init__(self):
        # 构造函数，加载检索问答链
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append(
                (question, self.chain({"query": question})["result"]))
            # 将问答结果直接附加到问答历史中，Gradio 会将其展示出来
            return "", chat_history
        except Exception as e:
            return e, chat_history


# 实例化核心功能对象
model_center = Model_center()
# 创建一个 Web 界面
block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            # 展示的页面标题
            gr.Markdown("""<h1><center>InternLMCooking</center></h1>
                <center>书生浦语-烹饪助手</center>
                """)
    with gr.Row():
        with gr.Column(scale=4):
            # 创建一个聊天机器人对象
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")
                
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])

    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)


gr.close_all()
# 直接启动
demo.launch()
