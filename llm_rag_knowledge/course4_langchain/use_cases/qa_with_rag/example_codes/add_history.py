#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project: llm-rag-knowledge
@File:    add_history
@Author:  yydsp
@Date:    2025/9/15 16:27
@Desc:
    1. rewrite_question: 重构问题节点。
    2. retrieve_documents: 检索文档节点。
    3. generate_answer: 生成答案节点。
"""

from pprint import pprint
from typing import List, TypedDict

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import StateGraph, END

# 初始化 LLM 和 嵌入模型
llm = ChatDeepSeek(
    model="deepseek-reasoner",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="sk-c1df83e7cea34495b96dd81cd6c606a2",
    # other params...
)
embeddings = OpenAIEmbeddings()

# 2. 创建向量数据库和检索器 (Retriever)
# 这是您问题的核心：`retriever` 是从哪里来的？
# 我们首先创建一些示例文档，然后将它们加载到一个向量数据库中（这里使用 FAISS）。
# 然后，我们从这个数据库中得到一个 retriever 对象。

docs = [
    Document(
        page_content="LangSmith is a platform for building production-grade LLM applications. It helps you trace, monitor, and debug LangChain applications."
    ),
    Document(
        page_content="LangSmith provides tools for testing and evaluating your LLM applications. You can create datasets, run tests, and visualize the results."
    ),
    Document(
        page_content="To use LangSmith, you need to set up an account and get an API key. Then, you can configure your environment variables to start logging traces."
    ),
    Document(
        page_content="LangChain is a framework for developing applications powered by language models. It provides modular components and ready-to-use chains."
    ),
]

# 从文档创建 FAISS 向量存储
vectorstore = FAISS.from_documents(docs, embedding=embeddings)

# retriever 对象就是向量存储的一个接口，它有一个 .invoke() 方法来执行搜索
retriever = vectorstore.as_retriever()

"""
from langchain_core.tools import tool

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    retrieved_docs = vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs
"""


# --- 向量库查询在这里发生 ---
# 在我们的 LangGraph 节点中，当我们调用 `retriever.invoke(question)` 时，
# 底层实际发生的是：
# 1. `question` 字符串被 embeddings 模型转换成一个向量。
# 2. FAISS 数据库执行一个高效的相似度搜索，用这个向量去匹配数据库中已有的文档向量。
# 3. 返回最相似的几个文档（Documents）。
# 所以，`retriever.invoke()` 就是您关心的“向量库查询”的动作。

# 3. 定义图的状态 (State)
class GraphState(TypedDict):
    """
    代表我们图的状态。

    Attributes:
        input: 用户的原始问题
        chat_history: 对话历史
        rewritten_question: 重构后的独立问题
        documents: 检索到的相关文档
        answer: 模型生成的最终答案
    """
    input: str
    chat_history: List[BaseMessage]
    rewritten_question: str
    documents: List[Document]
    answer: str


# 4. 定义图的节点 (Nodes)
# 增加一个新的节点，用于普通对话
def simple_chat(state: GraphState):
    """一个简单的聊天节点，不进行检索。"""
    print("--- 节点: simple_chat ---")

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful and friendly assistant. Respond to the user's greeting or question directly."),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ])

    chat_chain = prompt | llm | StrOutputParser()

    answer = chat_chain.invoke({
        "chat_history": state["chat_history"],
        "input": state["input"]
    })

    return {"answer": answer}


# 增加一个路由节点来决定走哪条路
def route_question(state: GraphState):
    """判断用户输入是需要检索(RAG)还是普通对话。"""
    print("--- 节点: route_question (决策中) ---")

    prompt = ChatPromptTemplate.from_string(
        """Given the user's input and the conversation history, classify the user's input.
Is it a question that requires retrieving documents from a knowledge base, or is it a simple greeting, chit-chat, or a question that can be answered without a knowledge base?

Return 'rag' for questions that need document retrieval.
Return 'chat' for simple greetings or chit-chat.

Conversation History:
{chat_history}

User Input:
{input}
"""
    )
    router_chain = prompt | llm | StrOutputParser()

    decision = router_chain.invoke({
        "chat_history": state["chat_history"],
        "input": state["input"]
    })

    print(f"路由决策: '{decision}'")
    if "rag" in decision.lower():
        return "rewrite"  # 返回下一个节点的名字
    else:
        return "chat"


def rewrite_question(state: GraphState):
    """根据对话历史重构问题，使其成为一个独立的问题。"""
    print("--- 节点: rewrite_question ---")
    prompt = ChatPromptTemplate.from_messages([
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    rewriter_chain = prompt | llm | StrOutputParser()
    rewritten_question = rewriter_chain.invoke({
        "chat_history": state["chat_history"],
        "input": state["input"]
    })
    print(f"重构后的问题: {rewritten_question}")
    return {"rewritten_question": rewritten_question}


def retrieve_documents(state: GraphState):
    """使用重构后的问题来检索文档。"""
    print("--- 节点: retrieve_documents ---")
    question = state["rewritten_question"]
    # ！！！向量库查询发生在这里！！！
    documents = retriever.invoke(question)
    print(f"检索到 {len(documents)} 个文档")
    return {"documents": documents}


def generate_answer(state: GraphState):
    """根据检索到的文档和原始问题生成最终答案。"""
    print("--- 节点: generate_answer ---")

    # 这里的 `create_stuff_documents_chain` 是一个从 langchain 库导入的函数，
    # 它本身会创建一个 Runnable (一个链)，而不是一个已经定义好的变量。
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        ("placeholder", "{chat_history}"),
        ("user", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, prompt)

    answer = document_chain.invoke({
        "context": state["documents"],
        "chat_history": state["chat_history"],
        "input": state["input"]
    })
    print("--- 生成最终答案 ---")
    return {"answer": answer}


# 5. 构建并编译图 (Graph)
workflow = StateGraph(GraphState)

# 添加所有需要的节点
workflow.add_node("route", route_question)
workflow.add_node("rewrite", rewrite_question)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)
workflow.add_node("chat", simple_chat)

# 设置入口点为新的路由节点
workflow.set_entry_point("route")

# 添加条件边！
# add_conditional_edges 的第一个参数是决策节点的名称。
# 第二个参数是决策函数（我们这里就是 route_question 节点本身）。
# 第三个参数是一个字典，将决策函数的返回值映射到下一个节点的名称。
workflow.add_conditional_edges(
    "route",
    route_question,
    {
        "rewrite": "rewrite",
        "chat": "chat"
    }
)

# 连接 RAG 路径的剩余部分
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "generate")

# 将两条路径的终点都连接到 END
workflow.add_edge("generate", END)
workflow.add_edge("chat", END)

# 编译新的图
app = workflow.compile()

if __name__ == '__main__':
    # 6. 执行多轮对话
    chat_history = []

    # 第一轮对话
    first_input = "Can LangSmith help test my LLM applications?"
    final_state = app.invoke({
        "chat_history": chat_history,
        "input": first_input
    })

    print("\n\n--- 对话第一轮结果 ---")
    pprint(final_state['answer'])

    # 更新聊天记录
    chat_history.extend([
        HumanMessage(content=first_input),
        AIMessage(content=final_state['answer'])
    ])
    """
    ### 问题一：为什么 `chat_history` 需要在外部手动维护？
    
    您观察得非常敏锐。`chat_history` 确实存在于 `GraphState` 中，但它只在**单次图的执行（a single run）**中存在。
    
    **核心原因：LangGraph 的每次 `invoke` 都是无状态的 (Stateless)。**
    
    可以把 `app.invoke()` 看作是一次**独立的、自包含的函数调用**。
    
    1.  **调用开始**：当你调用 `app.invoke({...})` 时，`LangGraph` 会根据你的输入字典创建一个全新的 `GraphState` 实例。
    2.  **执行期间**：这个 `GraphState` 实例会在图的各个节点之间传递，节点可以读取和修改它。
    3.  **调用结束**：当图运行到 `END` 时，这次执行就结束了。最终的状态被返回，然后这个 `GraphState` 实例就被**销毁**了。它不会被保存到下一次调用。
    """

    print("\n\n=========================\n\n")

    # 第二轮对话
    second_input = "Tell me how"
    final_state = app.invoke({
        "chat_history": chat_history,
        "input": second_input
    })

    print("\n\n--- 对话第二轮结果 ---")
    pprint(final_state['answer'])

    # 6.1 演示新的智能图

    chat_history = []

    # --- 第一次测试：简单的问候 ---
    print("****** 开始测试: 简单问候 ******")
    greeting_input = "Hi there, how are you?"
    final_state = app.invoke({
        "chat_history": chat_history,
        "input": greeting_input
    })
    print("\n--- 最终回复 ---")
    print(final_state['answer'])

    # 手动更新历史
    chat_history.extend([
        HumanMessage(content=greeting_input),
        AIMessage(content=final_state['answer'])
    ])

    print("\n\n=========================\n\n")

    # --- 第二次测试：需要 RAG 的问题 ---
    print("****** 开始测试: RAG 问题 ******")
    rag_input = "How can I use LangSmith to test my applications?"
    final_state = app.invoke({
        "chat_history": chat_history,
        "input": rag_input
    })
    print("\n--- 最终回复 ---")
    print(final_state['answer'])
