"""
LangChain 実践編 2: エージェント（Agents）

このサンプルでは、自律的にタスクを実行するエージェントの
作成方法を学びます。
"""

import os
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain import hub
import math

def main():
    """エージェントのデモ"""

    # 環境変数チェック
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: OPENAI_API_KEYが設定されていません")
        return

    print("=" * 50)
    print("LangChain 実践編 2: エージェント（Agents）")
    print("=" * 50)
    print()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # 1. カスタムツールの定義
    print("1. カスタムツールの定義")
    print("-" * 50)

    # 計算機ツール
    def calculator(expression: str) -> str:
        """数式を評価して結果を返します"""
        try:
            # 安全な評価のため、許可された関数のみ使用
            allowed_names = {
                k: v for k, v in math.__dict__.items()
                if not k.startswith("__")
            }
            allowed_names.update({
                "abs": abs,
                "round": round,
                "min": min,
                "max": max
            })
            result = eval(expression, {"__builtins__": {}}, allowed_names)
            return str(result)
        except Exception as e:
            return f"エラー: {str(e)}"

    # テキスト長カウンター
    def text_length(text: str) -> str:
        """テキストの長さを返します"""
        return f"テキストの長さ: {len(text)}文字"

    # 大文字変換ツール
    def to_uppercase(text: str) -> str:
        """テキストを大文字に変換します"""
        return text.upper()

    # リバースツール
    def reverse_text(text: str) -> str:
        """テキストを逆順にします"""
        return text[::-1]

    # ツールのリスト作成
    tools = [
        Tool(
            name="Calculator",
            func=calculator,
            description="数式を計算します。例: '2 + 2' や 'sqrt(16)'"
        ),
        Tool(
            name="TextLength",
            func=text_length,
            description="テキストの長さをカウントします"
        ),
        Tool(
            name="ToUppercase",
            func=to_uppercase,
            description="テキストを大文字に変換します"
        ),
        Tool(
            name="ReverseText",
            func=reverse_text,
            description="テキストを逆順にします"
        )
    ]

    print("定義されたツール:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")
    print()

    # 2. ReActエージェントの作成
    print("2. ReActエージェントの作成")
    print("-" * 50)

    # ReActプロンプトテンプレート
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""

    prompt = PromptTemplate.from_template(template)

    # エージェントの作成
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    print("ReActエージェント作成完了!")
    print()

    # 3. 基本的なエージェント実行
    print("3. 基本的なエージェント実行")
    print("-" * 50)

    result = agent_executor.invoke({
        "input": "25の平方根は何ですか？"
    })

    print(f"\n質問: {result['input']}")
    print(f"最終回答: {result['output']}")
    print()

    # 4. 複数ステップの推論
    print("4. 複数ステップの推論")
    print("-" * 50)

    result = agent_executor.invoke({
        "input": "「Hello World」という文字列を大文字に変換して、その長さを教えてください。"
    })

    print(f"\n質問: {result['input']}")
    print(f"最終回答: {result['output']}")
    print()

    # 5. より複雑なタスク
    print("5. より複雑なタスク")
    print("-" * 50)

    result = agent_executor.invoke({
        "input": "10と20の合計を計算し、その結果の平方根を求めてください。"
    })

    print(f"\n質問: {result['input']}")
    print(f"最終回答: {result['output']}")
    print()

    # 6. カスタムドメイン特化型エージェント
    print("6. カスタムドメイン特化型エージェント（商品在庫管理）")
    print("-" * 50)

    # 仮想の在庫データ
    inventory = {
        "laptop": {"stock": 15, "price": 1200},
        "mouse": {"stock": 50, "price": 25},
        "keyboard": {"stock": 30, "price": 75},
        "monitor": {"stock": 8, "price": 300}
    }

    def check_stock(product_name: str) -> str:
        """商品の在庫を確認します"""
        product = product_name.lower().strip()
        if product in inventory:
            stock = inventory[product]["stock"]
            return f"{product_name}の在庫: {stock}個"
        return f"{product_name}は商品リストにありません"

    def check_price(product_name: str) -> str:
        """商品の価格を確認します"""
        product = product_name.lower().strip()
        if product in inventory:
            price = inventory[product]["price"]
            return f"{product_name}の価格: ${price}"
        return f"{product_name}は商品リストにありません"

    def calculate_total(product_and_quantity: str) -> str:
        """商品と数量から合計金額を計算します。
        フォーマット: 'product:quantity' 例: 'laptop:3'
        """
        try:
            product, quantity = product_and_quantity.split(":")
            product = product.lower().strip()
            quantity = int(quantity.strip())

            if product in inventory:
                price = inventory[product]["price"]
                total = price * quantity
                return f"{product} x {quantity}個 = ${total}"
            return f"{product}は商品リストにありません"
        except Exception as e:
            return f"エラー: 正しいフォーマットで入力してください (product:quantity)"

    # 在庫管理ツール
    inventory_tools = [
        Tool(
            name="CheckStock",
            func=check_stock,
            description="商品の在庫数を確認します。商品名を入力してください。"
        ),
        Tool(
            name="CheckPrice",
            func=check_price,
            description="商品の価格を確認します。商品名を入力してください。"
        ),
        Tool(
            name="CalculateTotal",
            func=calculate_total,
            description="商品の合計金額を計算します。'商品名:数量'の形式で入力してください。例: 'laptop:3'"
        )
    ]

    # 在庫管理エージェント
    inventory_agent = create_react_agent(llm, inventory_tools, prompt)
    inventory_executor = AgentExecutor(
        agent=inventory_agent,
        tools=inventory_tools,
        verbose=True,
        handle_parsing_errors=True
    )

    # 在庫確認クエリ
    queries = [
        "laptopの在庫はいくつありますか？",
        "mouseを10個購入する場合、合計金額はいくらですか？",
        "monitorの価格と在庫を教えてください。"
    ]

    for query in queries:
        print(f"\nクエリ: {query}")
        print("-" * 50)
        result = inventory_executor.invoke({"input": query})
        print(f"回答: {result['output']}")
        print()

    # 7. エージェントの制限事項とベストプラクティス
    print("7. エージェントのベストプラクティス")
    print("-" * 50)
    print("""
    エージェント使用時のポイント:

    1. ツールの説明は明確に
       - ツールが何をするのか、どんな入力が必要かを明記

    2. エラーハンドリング
       - ツール関数内で適切なエラーハンドリングを実装

    3. ツール数の最適化
       - 必要最小限のツールを提供（多すぎるとLLMが混乱）

    4. verbose=True で動作確認
       - 開発中はverboseモードで推論プロセスを確認

    5. トークン制限に注意
       - 長い会話履歴は制限を超える可能性がある

    6. コスト管理
       - エージェントは複数回のLLM呼び出しを行うためコストに注意
    """)

    print("=" * 50)
    print("サンプル完了!")
    print("=" * 50)

if __name__ == "__main__":
    main()
