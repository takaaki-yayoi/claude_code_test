"""
LangChain 基本編 3: チェーン（Chains）

このサンプルでは、複数の処理を連鎖させるチェーンの
使い方を学びます。
"""

import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel

def main():
    """チェーンのデモ"""

    # 環境変数チェック
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: OPENAI_API_KEYが設定されていません")
        return

    print("=" * 50)
    print("LangChain 基本編 3: チェーン（Chains）")
    print("=" * 50)
    print()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 1. 基本的なLCEL（LangChain Expression Language）チェーン
    print("1. 基本的なLCELチェーン")
    print("-" * 50)

    prompt = ChatPromptTemplate.from_template(
        "{topic}について100文字程度で説明してください。"
    )

    # LCEL構文でチェーンを作成
    chain = prompt | llm | StrOutputParser()

    result = chain.invoke({"topic": "量子コンピューティング"})
    print(f"トピック: 量子コンピューティング")
    print(f"説明: {result}")
    print()

    # 2. 複数ステップのチェーン
    print("2. 複数ステップのチェーン")
    print("-" * 50)

    # ステップ1: トピックから質問を生成
    question_prompt = ChatPromptTemplate.from_template(
        "{topic}について初心者向けの質問を1つ作成してください。"
    )

    # ステップ2: 質問に答える
    answer_prompt = ChatPromptTemplate.from_template(
        "次の質問に答えてください: {question}"
    )

    # チェーンを組み合わせる
    question_chain = question_prompt | llm | StrOutputParser()
    answer_chain = answer_prompt | llm | StrOutputParser()

    # 実行
    topic = "機械学習"
    question = question_chain.invoke({"topic": topic})
    answer = answer_chain.invoke({"question": question})

    print(f"トピック: {topic}")
    print(f"生成された質問: {question}")
    print(f"回答: {answer}")
    print()

    # 3. 並列チェーン
    print("3. 並列チェーン（複数の処理を同時実行）")
    print("-" * 50)

    # 複数の異なる視点から分析
    technical_prompt = ChatPromptTemplate.from_template(
        "{concept}を技術的な観点から50文字程度で説明してください。"
    )
    business_prompt = ChatPromptTemplate.from_template(
        "{concept}をビジネス的な観点から50文字程度で説明してください。"
    )

    # 並列チェーンの作成
    parallel_chain = RunnableParallel(
        technical=technical_prompt | llm | StrOutputParser(),
        business=business_prompt | llm | StrOutputParser()
    )

    results = parallel_chain.invoke({"concept": "ブロックチェーン"})
    print(f"概念: ブロックチェーン")
    print(f"技術的視点: {results['technical']}")
    print(f"ビジネス視点: {results['business']}")
    print()

    # 4. 条件分岐を含むチェーン
    print("4. データ変換を含むチェーン")
    print("-" * 50)

    # 入力を変換してからLLMに渡す
    def uppercase_topic(inputs):
        """トピックを大文字に変換"""
        return {"topic": inputs["topic"].upper()}

    transform_chain = (
        RunnablePassthrough()
        | uppercase_topic
        | ChatPromptTemplate.from_template(
            "「{topic}」というキーワードに関連する技術を3つ挙げてください。"
        )
        | llm
        | StrOutputParser()
    )

    result = transform_chain.invoke({"topic": "ai"})
    print(f"元のトピック: ai")
    print(f"結果: {result}")
    print()

    # 5. 実践的な例：コード生成と説明チェーン
    print("5. 実践的な例：コード生成と説明チェーン")
    print("-" * 50)

    # コード生成プロンプト
    code_gen_prompt = ChatPromptTemplate.from_template(
        "{language}で{task}を実行する関数を作成してください。"
        "コードのみを出力してください。"
    )

    # コード説明プロンプト
    code_explain_prompt = ChatPromptTemplate.from_template(
        "次のコードを初心者向けに説明してください:\n\n{code}"
    )

    # チェーンの組み立て
    code_chain = code_gen_prompt | llm | StrOutputParser()

    # コード生成
    generated_code = code_chain.invoke({
        "language": "Python",
        "task": "フィボナッチ数列のn番目の数を計算"
    })

    # コード説明
    explain_chain = code_explain_prompt | llm | StrOutputParser()
    explanation = explain_chain.invoke({"code": generated_code})

    print(f"タスク: フィボナッチ数列のn番目の数を計算")
    print(f"\n生成されたコード:\n{generated_code}")
    print(f"\nコードの説明:\n{explanation}")
    print()

    # 6. カスタムチェーン関数
    print("6. カスタムチェーン関数")
    print("-" * 50)

    def create_summarize_and_translate_chain(target_language: str):
        """要約と翻訳を行うカスタムチェーン"""
        summarize_prompt = ChatPromptTemplate.from_template(
            "次のテキストを50文字程度に要約してください:\n{text}"
        )

        translate_prompt = ChatPromptTemplate.from_template(
            "次のテキストを{language}に翻訳してください:\n{summary}"
        )

        # 要約チェーン
        summarize = summarize_prompt | llm | StrOutputParser()

        # 翻訳チェーン（要約結果を使用）
        translate = (
            {
                "summary": lambda x: x,
                "language": lambda x: target_language
            }
            | translate_prompt
            | llm
            | StrOutputParser()
        )

        # チェーンの結合
        return summarize | translate

    chain = create_summarize_and_translate_chain("英語")

    long_text = """
    人工知能は、コンピュータサイエンスの一分野であり、
    人間の知能を模倣する機械を作成することを目指しています。
    機械学習、深層学習、自然言語処理などの技術が含まれ、
    画像認識、音声認識、自動運転など様々な応用があります。
    """

    result = chain.invoke({"text": long_text})
    print(f"元のテキスト: {long_text.strip()}")
    print(f"要約＆翻訳結果: {result}")
    print()

    print("=" * 50)
    print("サンプル完了!")
    print("=" * 50)

if __name__ == "__main__":
    main()
