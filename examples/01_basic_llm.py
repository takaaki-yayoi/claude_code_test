"""
LangChain 基本編 1: LLMの基本的な使い方

このサンプルでは、LangChainを使って言語モデルを呼び出す
基本的な方法を学びます。
"""

import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage

def main():
    """LLMの基本的な使い方のデモ"""

    # 環境変数からAPIキーを取得
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("エラー: OPENAI_API_KEYが設定されていません")
        print("export OPENAI_API_KEY='your-api-key' を実行してください")
        return

    print("=" * 50)
    print("LangChain 基本編 1: LLMの基本的な使い方")
    print("=" * 50)
    print()

    # 1. ChatOpenAIモデルの初期化
    print("1. ChatOpenAIモデルを初期化...")
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,  # 0に近いほど決定的、1に近いほどランダム
        max_tokens=500
    )
    print("   初期化完了!")
    print()

    # 2. 単純なテキスト生成
    print("2. 単純なテキスト生成の例")
    print("-" * 50)
    messages = [
        HumanMessage(content="こんにちは！LangChainについて簡単に教えてください。")
    ]
    response = llm.invoke(messages)
    print(f"質問: {messages[0].content}")
    print(f"回答: {response.content}")
    print()

    # 3. システムメッセージを使った例
    print("3. システムメッセージを使った例")
    print("-" * 50)
    messages = [
        SystemMessage(content="あなたは親切なプログラミング講師です。初心者にもわかりやすく説明してください。"),
        HumanMessage(content="Pythonのリスト内包表記について教えてください。")
    ]
    response = llm.invoke(messages)
    print(f"システムメッセージ: {messages[0].content}")
    print(f"質問: {messages[1].content}")
    print(f"回答: {response.content}")
    print()

    # 4. 複数のメッセージを使った会話
    print("4. 複数のメッセージを使った会話の例")
    print("-" * 50)
    messages = [
        SystemMessage(content="あなたは優秀なAIアシスタントです。"),
        HumanMessage(content="JavaScriptとPythonの違いは何ですか？"),
    ]
    response = llm.invoke(messages)
    print(f"質問: {messages[1].content}")
    print(f"回答: {response.content}")
    print()

    # 5. 温度パラメータの違い
    print("5. 温度パラメータの違いを確認")
    print("-" * 50)

    # 低い温度（決定的）
    llm_low_temp = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)
    messages = [HumanMessage(content="1+1は？")]
    response_low = llm_low_temp.invoke(messages)
    print(f"低温度(0.0)の回答: {response_low.content}")

    # 高い温度（創造的）
    llm_high_temp = ChatOpenAI(model="gpt-3.5-turbo", temperature=1.0)
    messages = [HumanMessage(content="空について詩的に表現してください。")]
    response_high = llm_high_temp.invoke(messages)
    print(f"高温度(1.0)の回答: {response_high.content}")
    print()

    print("=" * 50)
    print("サンプル完了!")
    print("=" * 50)

if __name__ == "__main__":
    main()
