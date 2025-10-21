"""
LangChain 実践編 1: RAG（Retrieval-Augmented Generation）

このサンプルでは、ドキュメント検索と生成を組み合わせた
RAGシステムの構築方法を学びます。
"""

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def main():
    """RAGシステムのデモ"""

    # 環境変数チェック
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: OPENAI_API_KEYが設定されていません")
        return

    print("=" * 50)
    print("LangChain 実践編 1: RAG（Retrieval-Augmented Generation）")
    print("=" * 50)
    print()

    # LLMとEmbeddingsの初期化
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    embeddings = OpenAIEmbeddings()

    # 1. サンプルドキュメントの準備
    print("1. サンプルドキュメントの準備")
    print("-" * 50)

    documents = [
        """
        LangChainは、大規模言語モデル（LLM）を活用したアプリケーション開発を
        支援するフレームワークです。Pythonで書かれており、様々なLLMプロバイダーと
        統合することができます。主な機能には、プロンプト管理、チェーン、
        エージェント、メモリ管理などがあります。
        """,
        """
        RAG（Retrieval-Augmented Generation）は、検索技術と生成AIを組み合わせた
        手法です。まず関連するドキュメントを検索し、その情報を元にLLMが回答を
        生成します。これにより、LLMの知識を拡張し、より正確で最新の情報を
        提供できます。
        """,
        """
        ベクトルデータベースは、テキストや画像などのデータを高次元ベクトルとして
        保存し、類似度検索を高速に行うためのデータベースです。FAISSやChroma、
        Pineconeなどが代表的です。RAGシステムにおいて、関連ドキュメントの検索に
        使用されます。
        """,
        """
        エンベディング（Embedding）は、テキストを数値ベクトルに変換する技術です。
        意味的に似たテキストは、ベクトル空間で近い位置に配置されます。
        OpenAIのtext-embedding-ada-002などのモデルが広く使用されています。
        """,
        """
        プロンプトエンジニアリングは、LLMから望ましい出力を得るために、
        入力プロンプトを最適化する技術です。Few-shot学習、Chain-of-Thought、
        役割設定などの手法があります。効果的なプロンプト設計により、
        LLMのパフォーマンスを大幅に向上させることができます。
        """
    ]

    print(f"準備されたドキュメント数: {len(documents)}")
    print()

    # 2. ドキュメントの分割
    print("2. ドキュメントの分割")
    print("-" * 50)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,  # チャンクのサイズ
        chunk_overlap=50,  # チャンク間のオーバーラップ
        separators=["\n\n", "\n", "。", " ", ""]
    )

    texts = []
    for doc in documents:
        splits = text_splitter.split_text(doc.strip())
        texts.extend(splits)

    print(f"分割後のチャンク数: {len(texts)}")
    print(f"最初のチャンク例:\n{texts[0]}")
    print()

    # 3. ベクトルストアの作成
    print("3. ベクトルストアの作成（FAISSを使用）")
    print("-" * 50)

    print("エンベディングを生成してベクトルストアを作成中...")
    vectorstore = FAISS.from_texts(texts, embeddings)
    print("ベクトルストアの作成完了!")
    print()

    # 4. 類似度検索のデモ
    print("4. 類似度検索のデモ")
    print("-" * 50)

    query = "RAGとは何ですか？"
    docs = vectorstore.similarity_search(query, k=2)

    print(f"クエリ: {query}")
    print(f"検索結果（上位2件）:")
    for i, doc in enumerate(docs, 1):
        print(f"\n  結果{i}: {doc.page_content[:100]}...")
    print()

    # 5. 基本的なRAGチェーン
    print("5. 基本的なRAGチェーン")
    print("-" * 50)

    # RetrievalQAチェーンの作成
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )

    query = "LangChainの主な機能は何ですか？"
    result = qa_chain.invoke({"query": query})

    print(f"質問: {query}")
    print(f"回答: {result['result']}")
    print()

    # 6. カスタムプロンプトを使用したRAG
    print("6. カスタムプロンプトを使用したRAG")
    print("-" * 50)

    template = """以下のコンテキスト情報を使用して、質問に答えてください。
    わからない場合は、わからないと言ってください。無理に答えを作らないでください。

    コンテキスト:
    {context}

    質問: {question}

    回答:"""

    prompt = ChatPromptTemplate.from_template(template)

    # カスタムRAGチェーンの構築
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    rag_chain = (
        {
            "context": retriever | (lambda docs: "\n\n".join([d.page_content for d in docs])),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    questions = [
        "エンベディングとは何ですか？",
        "ベクトルデータベースの用途は？",
        "プロンプトエンジニアリングの手法を教えてください。"
    ]

    for question in questions:
        answer = rag_chain.invoke(question)
        print(f"Q: {question}")
        print(f"A: {answer}")
        print()

    # 7. メタデータフィルタリング付きRAG
    print("7. メタデータを含むRAGの例")
    print("-" * 50)

    # メタデータ付きドキュメント
    from langchain.schema import Document

    docs_with_metadata = [
        Document(
            page_content=text,
            metadata={"source": f"doc_{i}", "category": "tutorial"}
        )
        for i, text in enumerate(texts)
    ]

    vectorstore_with_metadata = FAISS.from_documents(
        docs_with_metadata,
        embeddings
    )

    # メタデータを含む検索
    query = "LangChainについて"
    results = vectorstore_with_metadata.similarity_search(query, k=2)

    print(f"クエリ: {query}")
    for i, doc in enumerate(results, 1):
        print(f"\n結果{i}:")
        print(f"  内容: {doc.page_content[:80]}...")
        print(f"  メタデータ: {doc.metadata}")
    print()

    # 8. 実践的な例：FAQシステム
    print("8. 実践的な例：FAQシステム")
    print("-" * 50)

    faq_data = [
        {
            "question": "LangChainのインストール方法は？",
            "answer": "pip install langchain でインストールできます。"
        },
        {
            "question": "どのLLMが対応していますか？",
            "answer": "OpenAI、Anthropic、Google、Hugging Faceなど多数のLLMに対応しています。"
        },
        {
            "question": "商用利用は可能ですか？",
            "answer": "LangChain自体はMITライセンスで商用利用可能ですが、使用するLLMのライセンスに従う必要があります。"
        }
    ]

    # FAQをドキュメント化
    faq_texts = [f"Q: {item['question']}\nA: {item['answer']}" for item in faq_data]
    faq_vectorstore = FAISS.from_texts(faq_texts, embeddings)

    # FAQチェーン
    faq_chain = (
        {
            "context": faq_vectorstore.as_retriever() | (lambda docs: "\n\n".join([d.page_content for d in docs])),
            "question": RunnablePassthrough()
        }
        | ChatPromptTemplate.from_template(
            "以下のFAQを参考に質問に答えてください:\n\n{context}\n\n質問: {question}"
        )
        | llm
        | StrOutputParser()
    )

    user_question = "LangChainはどうやって使い始めればいいですか？"
    answer = faq_chain.invoke(user_question)

    print(f"ユーザーの質問: {user_question}")
    print(f"回答: {answer}")
    print()

    print("=" * 50)
    print("サンプル完了!")
    print("=" * 50)

if __name__ == "__main__":
    main()
