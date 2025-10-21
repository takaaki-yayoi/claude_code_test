"""
LangChain 基本編 2: プロンプトテンプレート

このサンプルでは、再利用可能なプロンプトテンプレートの
作成方法を学びます。
"""

import os
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    FewShotPromptTemplate
)

def main():
    """プロンプトテンプレートのデモ"""

    # 環境変数チェック
    if not os.getenv("OPENAI_API_KEY"):
        print("エラー: OPENAI_API_KEYが設定されていません")
        return

    print("=" * 50)
    print("LangChain 基本編 2: プロンプトテンプレート")
    print("=" * 50)
    print()

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

    # 1. 基本的なPromptTemplate
    print("1. 基本的なPromptTemplate")
    print("-" * 50)

    template = "次の{language}のコードを説明してください: {code}"
    prompt = PromptTemplate(
        input_variables=["language", "code"],
        template=template
    )

    formatted_prompt = prompt.format(
        language="Python",
        code="list(map(lambda x: x**2, range(10)))"
    )
    print(f"フォーマット済みプロンプト:\n{formatted_prompt}")
    print()

    # 2. ChatPromptTemplate
    print("2. ChatPromptTemplate（複数のメッセージ）")
    print("-" * 50)

    system_template = "あなたは{expertise}の専門家です。"
    human_template = "{question}"

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])

    messages = chat_prompt.format_messages(
        expertise="機械学習",
        question="ニューラルネットワークの基本的な仕組みを教えてください。"
    )

    response = llm.invoke(messages)
    print(f"質問: ニューラルネットワークの基本的な仕組みを教えてください。")
    print(f"回答: {response.content}")
    print()

    # 3. Few-Shot プロンプトテンプレート
    print("3. Few-Shot プロンプトテンプレート")
    print("-" * 50)

    # サンプル例
    examples = [
        {"word": "happy", "antonym": "sad"},
        {"word": "tall", "antonym": "short"},
        {"word": "hot", "antonym": "cold"}
    ]

    # 例のフォーマット
    example_template = """
    単語: {word}
    反対語: {antonym}
    """

    example_prompt = PromptTemplate(
        input_variables=["word", "antonym"],
        template=example_template
    )

    # Few-shotプロンプトの作成
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="以下は単語とその反対語の例です:",
        suffix="\n単語: {input}\n反対語:",
        input_variables=["input"]
    )

    formatted = few_shot_prompt.format(input="big")
    print(f"Few-shotプロンプト:\n{formatted}")
    print()

    # 4. 実践的な例：コードレビュー用テンプレート
    print("4. 実践的な例：コードレビュー用テンプレート")
    print("-" * 50)

    review_template = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "あなたは経験豊富な{language}開発者です。"
            "コードレビューを行い、改善点を提案してください。"
        ),
        HumanMessagePromptTemplate.from_template(
            "以下のコードをレビューしてください:\n\n{code}\n\n"
            "特に{focus}に注目してください。"
        )
    ])

    code_sample = """
def calculate_average(numbers):
    total = 0
    for num in numbers:
        total = total + num
    return total / len(numbers)
    """

    messages = review_template.format_messages(
        language="Python",
        code=code_sample,
        focus="効率性とエラーハンドリング"
    )

    response = llm.invoke(messages)
    print(f"レビュー対象コード:\n{code_sample}")
    print(f"レビュー結果:\n{response.content}")
    print()

    # 5. 条件付きプロンプト
    print("5. 動的なプロンプト生成")
    print("-" * 50)

    def create_translation_prompt(formal: bool = False):
        """フォーマルさに応じて異なるプロンプトを生成"""
        style = "敬語を使って丁寧に" if formal else "カジュアルに"

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                f"あなたは翻訳者です。{style}翻訳してください。"
            ),
            HumanMessagePromptTemplate.from_template(
                "{source_language}から{target_language}に翻訳してください:\n{text}"
            )
        ])

    # カジュアルな翻訳
    casual_prompt = create_translation_prompt(formal=False)
    messages = casual_prompt.format_messages(
        source_language="英語",
        target_language="日本語",
        text="How are you doing today?"
    )
    response = llm.invoke(messages)
    print(f"カジュアル翻訳: {response.content}")

    # フォーマルな翻訳
    formal_prompt = create_translation_prompt(formal=True)
    messages = formal_prompt.format_messages(
        source_language="英語",
        target_language="日本語",
        text="How are you doing today?"
    )
    response = llm.invoke(messages)
    print(f"フォーマル翻訳: {response.content}")
    print()

    print("=" * 50)
    print("サンプル完了!")
    print("=" * 50)

if __name__ == "__main__":
    main()
