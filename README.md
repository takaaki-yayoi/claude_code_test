# LangChain チュートリアル

このリポジトリはLangChainの基本的な使い方から実践的な応用まで学べるチュートリアルです。

## 目次

1. [環境構築](#環境構築)
2. [基本編](#基本編)
3. [実践編](#実践編)
4. [サンプルコード](#サンプルコード)

## 環境構築

### 必要なパッケージのインストール

```bash
pip install -r requirements.txt
```

### 環境変数の設定

OpenAI APIキーを設定してください：

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## 基本編

### 1. LLMの基本的な使い方

LangChainを使って言語モデルを呼び出す基本的な方法を学びます。

- **ファイル**: `examples/01_basic_llm.py`
- **学習内容**: LLMの初期化、テキスト生成の基本

### 2. プロンプトテンプレート

再利用可能なプロンプトテンプレートの作成方法を学びます。

- **ファイル**: `examples/02_prompt_templates.py`
- **学習内容**: テンプレートの作成、変数の埋め込み、Few-shotプロンプト

### 3. チェーン（Chains）

複数の処理を連鎖させるチェーンの使い方を学びます。

- **ファイル**: `examples/03_chains.py`
- **学習内容**: LLMChain、SequentialChain、カスタムチェーン

## 実践編

### 4. RAG（Retrieval-Augmented Generation）

ドキュメント検索と生成を組み合わせたRAGシステムの構築方法を学びます。

- **ファイル**: `examples/04_rag_example.py`
- **学習内容**:
  - ドキュメントの読み込みと分割
  - ベクトルストアの作成
  - 類似度検索
  - QAチェーンの構築

### 5. エージェント（Agents）

自律的にタスクを実行するエージェントの作成方法を学びます。

- **ファイル**: `examples/05_agent_example.py`
- **学習内容**:
  - ツールの定義
  - エージェントの初期化
  - タスクの実行

## サンプルコードの実行方法

各サンプルは独立して実行できます：

```bash
# 基本的なLLMの使い方
python examples/01_basic_llm.py

# プロンプトテンプレート
python examples/02_prompt_templates.py

# チェーン
python examples/03_chains.py

# RAGの例
python examples/04_rag_example.py

# エージェントの例
python examples/05_agent_example.py
```

## 学習のポイント

1. **段階的に学習**: 基本編から順番に進めることをお勧めします
2. **実際に動かす**: サンプルコードを実行して動作を確認しましょう
3. **コードを改変**: サンプルコードを自分で改変して理解を深めましょう
4. **ドキュメント参照**: [LangChain公式ドキュメント](https://python.langchain.com/)も併せて参照してください

## トラブルシューティング

### よくある問題

1. **APIキーのエラー**
   - 環境変数が正しく設定されているか確認してください
   - APIキーが有効か確認してください

2. **パッケージのインポートエラー**
   - `pip install -r requirements.txt` を再度実行してください
   - Python 3.8以上を使用していることを確認してください

3. **レート制限エラー**
   - API呼び出しの頻度を下げてください
   - 有料プランへのアップグレードを検討してください

## 参考資料

- [LangChain公式ドキュメント](https://python.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [OpenAI API ドキュメント](https://platform.openai.com/docs)

## ライセンス

このチュートリアルはMITライセンスの下で公開されています。
