#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@Project: llm-rag-knowledge
@File:    stream_out
@Author:  yydsp
@Date:    2025/9/15 17:58
@Desc: 
    
"""
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 关键点：选择一个支持流式输出的 LLM
# ChatOpenAI, ChatGoogleGenerativeAI, ChatAnthropic 等主流模型都支持
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 创建检索器 (与之前完全相同)
embeddings = OpenAIEmbeddings()
docs = ...  # 省略示例文档
vectorstore = FAISS.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever()

# 创建文档处理链
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)

# 创建检索链
# 注意：这个 retrieval_chain 既可以用于 .invoke() 也可以用于 .stream()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 创建文档处理链
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")
document_chain = create_stuff_documents_chain(llm, prompt)

# 创建检索链
# 注意：这个 retrieval_chain 既可以用于 .invoke() 也可以用于 .stream()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.stream({"input": "how can langsmith help with testing?"})

for chunk in response:
    print(chunk)
