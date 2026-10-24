# 10. RAG 후보 비교, 릴리스 활성화와 rollback 실습

목표는 RAG 도구를 하나씩 실행하는 것이 아니라, **같은 문서·검색 설정·프롬프트를 사용하는 후보를 검증하고 실제 질의 경로에 배포하는 것**입니다. 이 장의 기본 과정은 인터넷 API 호출, 클라우드 리소스 생성과 유료 LLM 없이 실행됩니다.

## 준비와 산출물

Python 3.11~3.12, uv와 저장소가 필요합니다. 모든 명령은 프로젝트 루트의 PowerShell 기준입니다.

```powershell
uv sync --extra dev
uv run ml-labs rag release init-demo --output .ml-labs\rag-demo\inputs
Get-ChildItem .ml-labs\rag-demo\inputs
```

입력 디렉터리에는 다음 네 파일이 생깁니다.

- `documents.jsonl`: 직접 작성한 공개 합성 MLOps 문서
- `config.json`: TF-IDF 검색과 응답 provider 설정
- `prompt.json`: 이름·버전·템플릿이 있는 로컬 프롬프트
- `evaluation.json`: 답변 가능 사례 두 개와 근거 없는 질문 한 개

이미 존재하는 디렉터리를 덮어쓰지 않습니다. 재실습할 때는 `rag-demo2`처럼 새 경로를 사용합니다.

## 1. 입력을 읽고 변경 지점을 이해하기

```powershell
Get-Content .ml-labs\rag-demo\inputs\config.json
Get-Content .ml-labs\rag-demo\inputs\prompt.json
Get-Content .ml-labs\rag-demo\inputs\evaluation.json
```

기본 설정은 다음과 같습니다.

```json
{
  "retriever": "tfidf",
  "top_k": 2,
  "min_score": 0.15,
  "ngram_range": [1, 2],
  "stop_words": "english",
  "provider": "extractive_offline"
}
```

`top_k`는 검색 결과 수의 상한이고 `min_score`는 최소 코사인 유사도입니다. 점수가 0인 문서는 임계값이 0이어도 제외합니다. n-gram과 stop words도 후보의 해시에 포함됩니다. 현재 검색기는 로컬 TF-IDF뿐이며 벡터 임베딩 모델이나 하이브리드 검색을 사용하지 않습니다.

프롬프트 템플릿에는 `{context}`와 `{question}`이 반드시 있어야 합니다. 로컬 모드는 문서를 인용할 뿐 템플릿으로 새 문장을 생성하지 않습니다. 프롬프트를 바꾸어도 로컬 답변의 문체가 바뀌지 않는 것이 정상입니다. Azure 모드를 선택하면 이 고정 템플릿이 실제 생성 요청에 사용됩니다.

평가 사례의 계약은 다음과 같습니다.

```json
[
  {
    "id": "registry",
    "query": "model registry champion promotion",
    "expected_ids": ["doc-mlflow-registry"],
    "answer_contains": ["validation"]
  },
  {
    "id": "unknown",
    "query": "quasar teleportation warranty",
    "unanswerable": true
  }
]
```

답변 가능 사례에는 `expected_ids`가 필요합니다. `answer_contains`는 선택 사항이며 대소문자를 무시한 부분 문자열 검사입니다. 거절 사례에는 기대 인용이나 답변 문자열을 지정하지 않습니다. 둘 중 한 종류만 있으면 후보 생성을 거부합니다. **위 예시는 형식 설명이며, 실제 데모 파일에는 feature store 사례도 포함됩니다.**

## 2. v1 후보를 고정하고 첫 평가하기

```powershell
uv run ml-labs rag release create --corpus .ml-labs\rag-demo\inputs\documents.jsonl --config .ml-labs\rag-demo\inputs\config.json --prompt .ml-labs\rag-demo\inputs\prompt.json --evaluation .ml-labs\rag-demo\inputs\evaluation.json --output .ml-labs\rag-demo\v1
uv run ml-labs rag release evaluate --candidate .ml-labs\rag-demo\v1\candidate.json --report .ml-labs\rag-demo\v1-report.json
$v1Report = Get-Content .ml-labs\rag-demo\v1-report.json -Raw | ConvertFrom-Json
$v1Report.passed
$v1Report.summary
```

기본 데모는 `passed: true`, `pass_rate: 1`이어야 합니다. 사례별 응답에는 검색 문서, 인용, 거절 여부, 후보 ID, 프롬프트 버전과 실제 측정된 지연이 있습니다.

```powershell
$v1Report.cases | Select-Object id, passed, baseline_passed
$v1Report.hashes
$v1Report.cases[0].response.metrics
```

보고서는 설정·문서·프롬프트·평가 집합의 해시와 `candidate_id`를 포함합니다. 보고서 자체에도 `report_id` 해시가 있습니다. 숫자만 적힌 다른 보고서를 붙이거나 후보 파일을 편집하면 활성화 시 검증에 실패합니다.

이 검사는 로컬 파일의 변경을 탐지하는 기능입니다. 서명이나 제삼자의 증명은 아니므로 파일을 쓰는 사람을 신뢰해야 합니다. 운영 환경에서는 별도 권한과 서명·감사를 추가해야 합니다.

## 3. 통과한 v1을 활성화하고 조회하기

```powershell
uv run ml-labs rag release activate --store .ml-labs\rag-demo\store --candidate .ml-labs\rag-demo\v1\candidate.json --report .ml-labs\rag-demo\v1-report.json
$v1 = (Get-Content .ml-labs\rag-demo\store\active.json -Raw | ConvertFrom-Json).active
uv run ml-labs rag release query --store .ml-labs\rag-demo\store --query "model registry champion promotion"
uv run ml-labs rag release query --store .ml-labs\rag-demo\store --query "quasar teleportation warranty"
```

첫 질의에는 `doc-mlflow-registry` 인용이 있어야 합니다. 두 번째 질의는 다음 상태를 반환해야 합니다.

```json
{
  "status": "no_grounding",
  "answer": "No grounded answer is available in the selected corpus.",
  "citations": [],
  "abstained": true,
  "citations_valid": true
}
```

실제 응답에는 `release_id`, `candidate_id`, 질의별 `trace_id`와 `metrics`도 포함됩니다. 배포는 `releases\<release_id>`에 검증된 후보와 보고서를 복사한 뒤 `active.json`을 같은 파일시스템에서 원자적으로 교체합니다. 준비 도중 오류가 나면 활성 포인터는 바뀌지 않습니다. 쓰기 작업은 로컬 잠금 파일로 직렬화합니다.

장애로 프로세스가 강제 종료되어 `.release.lock`이 남았다면 **실제 다른 작성자가 없는지 확인한 뒤에만** 잠금을 수동 정리합니다. NFS·여러 서버를 위한 분산 릴리스 관리 시스템은 아닙니다.

## 4. 나빠진 검색 설정을 만들고 회귀를 차단하기

`min_score`를 지나치게 높이면 관련 문서가 있어도 검색하지 못합니다. 원본과 v1 스냅샷은 건드리지 않고 새 설정을 만듭니다.

```powershell
$badConfig = Get-Content .ml-labs\rag-demo\inputs\config.json -Raw | ConvertFrom-Json
$badConfig.min_score = 1.0
$badConfig | ConvertTo-Json -Depth 10 | Set-Content .ml-labs\rag-demo\bad-config.json -Encoding utf8
uv run ml-labs rag release create --corpus .ml-labs\rag-demo\inputs\documents.jsonl --config .ml-labs\rag-demo\bad-config.json --prompt .ml-labs\rag-demo\inputs\prompt.json --evaluation .ml-labs\rag-demo\inputs\evaluation.json --output .ml-labs\rag-demo\bad
uv run ml-labs rag release evaluate --candidate .ml-labs\rag-demo\bad\candidate.json --baseline .ml-labs\rag-demo\v1\candidate.json --min-pass-rate 0.3 --max-regression 0 --report .ml-labs\rag-demo\bad-report.json
$bad = Get-Content .ml-labs\rag-demo\bad-report.json -Raw | ConvertFrom-Json
$bad.passed
$bad.summary
uv run ml-labs rag release activate --store .ml-labs\rag-demo\store --candidate .ml-labs\rag-demo\bad\candidate.json --report .ml-labs\rag-demo\bad-report.json
```

最後の activate は**失敗するのが期待動作**です。`min-pass-rate`を0.3に緩めても、v1が答えていた二つのケースが失敗するので回帰ゲートで止まります。`active.json`の値は `$v1`のままです。

回帰許容量 `max_regression` は、**baselineが通過し候補が失敗したケース数 ÷ 全ケース数**です。別の質問が改善しても、失ったケースを相殺しません。baselineは候補と同じ評価質問で実行するので、評価データを別々に採点して平均だけ比較するのとは異なります。

`--max-latency-ms`を指定すれば、その実行で測った候補のp95にも上限を置けます。

```powershell
uv run ml-labs rag release evaluate --candidate .ml-labs\rag-demo\v1\candidate.json --baseline .ml-labs\rag-demo\v1\candidate.json --max-latency-ms 1000 --report .ml-labs\rag-demo\v1-latency-report.json
```

1000msは説明用の予算であり、この環境での性能保証ではありません。少数ケースのローカル測定は負荷試験ではありません。モデル生成や外部評価の費用上限は、このレポートから実際の請求額を取得できないため実装していません。ゲート不合格でも報告書は保存され、CLIのevaluateは非ゼロで終了します。

## 5. v2を通して rollback する

まず、文書の意味を変えない検索設定の変更を試します。

```powershell
$v2Config = Get-Content .ml-labs\rag-demo\inputs\config.json -Raw | ConvertFrom-Json
$v2Config.top_k = 3
$v2Config | ConvertTo-Json -Depth 10 | Set-Content .ml-labs\rag-demo\v2-config.json -Encoding utf8
uv run ml-labs rag release create --corpus .ml-labs\rag-demo\inputs\documents.jsonl --config .ml-labs\rag-demo\v2-config.json --prompt .ml-labs\rag-demo\inputs\prompt.json --evaluation .ml-labs\rag-demo\inputs\evaluation.json --output .ml-labs\rag-demo\v2
uv run ml-labs rag release evaluate --candidate .ml-labs\rag-demo\v2\candidate.json --baseline .ml-labs\rag-demo\v1\candidate.json --report .ml-labs\rag-demo\v2-report.json
uv run ml-labs rag release activate --store .ml-labs\rag-demo\store --candidate .ml-labs\rag-demo\v2\candidate.json --report .ml-labs\rag-demo\v2-report.json
$v2 = (Get-Content .ml-labs\rag-demo\store\active.json -Raw | ConvertFrom-Json).active
uv run ml-labs rag release query --store .ml-labs\rag-demo\store --query "Feast offline online feature store"
uv run ml-labs rag release rollback --store .ml-labs\rag-demo\store
uv run ml-labs rag release query --store .ml-labs\rag-demo\store --query "Feast offline online feature store"
```

rollback後の応答の `release_id` は `$v1`です。明示的に既知のリリースを指定することもできます。

```powershell
uv run ml-labs rag release query --store .ml-labs\rag-demo\store --release-id $v2 --query "model registry champion"
uv run ml-labs rag release rollback --store .ml-labs\rag-demo\store --release-id $v2
```

一度も有効化していないIDへのrollbackや、ゲート不合格の候補への切り替えはできません。元の候補ディレクトリを削除してもリリースのコピーから実行できます。ただし稼働中の `store` 自体は削除しないでください。

## 6. 同じリリースをHTTPから使う

別のPowerShellでサーバーを起動します。

```powershell
uv run ml-labs serve --rag-release .ml-labs\rag-demo\store --host 127.0.0.1 --port 8000
```

クライアント側で送信します。

```powershell
$body = @{ query = "model registry champion promotion" } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/rag/query -Method Post -ContentType "application/json" -Body $body
Invoke-WebRequest http://127.0.0.1:8000/metrics | Select-Object -ExpandProperty Content
```

CLIと同じ `release_id`、候補ID、インデックス設定で応答することを確認します。APIはアクティブポインターをリクエストごとに一度解決します。別のシェルでrollbackしてから再送すれば、新しいリクエストから切り替わります。これは文書認可のある本番APIではなく、公開合成データ用のローカル演習です。

## 7. プロンプトの版を管理する

CLIでテンプレート文字列を登録し、同じ名前・版の内容変更を拒否します。`--template`はファイルパスではなく、テンプレート本文です。

```powershell
uv run ml-labs rag prompt register --store .ml-labs\rag-demo\store --name grounded-answer --version 2 --template "Use only the passages: {context}. Question: {question}. Return JSON with answer, citations, abstained. Cite every passage as [ID]; abstain if unsupported."
uv run ml-labs rag release create --corpus .ml-labs\rag-demo\inputs\documents.jsonl --config .ml-labs\rag-demo\inputs\config.json --prompt .ml-labs\rag-demo\store\prompts\grounded-answer\2.json --evaluation .ml-labs\rag-demo\inputs\evaluation.json --output .ml-labs\rag-demo\prompt-v2
```

新しいプロンプトにも評価・ゲート・activateが必要です。ローカル引用モードでは文面に影響しませんが、版とハッシュは変わり、Azure生成では実際の入力が変わります。この登録先はローカルJSONファイルであり、Langfuseネイティブのプロンプト管理ではありません。

## 8. ネイティブPromptfooで同じ候補を評価する

独立した固定プロンプトではなく、実際の候補を呼ぶPython provider設定をCLIで生成します。Node.jsが必要です。以下は通常の `.venv` を使った基本手順の続きであり、専用環境を選んだ場合は `PROMPTFOO_PYTHON` のパスも合わせます。

```powershell
uv run ml-labs tools promptfoo release-config --candidate .ml-labs\rag-demo\v1\candidate.json --output .ml-labs\rag-demo\promptfoo-release.yaml
$env:PROMPTFOO_PYTHON = (Resolve-Path .venv\Scripts\python.exe).Path
$env:PROMPTFOO_DISABLE_TELEMETRY = "1"
npm exec --yes --package=promptfoo@0.125.0 -- promptfoo eval --config .ml-labs\rag-demo\promptfoo-release.yaml --no-cache --output .ml-labs\rag-demo\promptfoo-results.json
$promptfoo = Get-Content .ml-labs\rag-demo\promptfoo-results.json -Raw | ConvertFrom-Json
$promptfoo.results.stats | Select-Object successes, failures, errors
```

バージョンを `0.125.0` に固定して実行します。初回はnpmパッケージの取得で通信しますが、`extractive_offline`候補の評価に有料モデルAPIは不要です。`--no-cache`により実際のRAGを再実行します。生成設定にはPython実行ファイルも記録されるので、環境を変更したら設定を作り直してください。

実装検証ではこのネイティブCLIで、registry・feature store・根拠なしの**3件がすべて成功し、失敗0・provider error 0**でした。JavaScript assertionは文字列JSONとオブジェクトの両方を扱える単一式です。Promptfooのrequest数は評価呼び出し数を示す場合があり、有料LLM呼び出し数とは別です。実際の `response.metrics.llm_calls` を確認してください。

Promptfoo報告は補助の回帰結果です。activateで必要なプロジェクト固有の `rag release evaluate` 報告書を置き換えるものではありません。

## 9. DeepEvalとLangfuseを任意で追加する

**ここからは外部サービスを使う任意経路**です。以下は実行手順であり、今回のローカル検証ではDeepEval評価者やLangfuseへの送信は実行していません。

DeepEvalを選ぶと、実際の回答と検索文脈でAnswer RelevancyとFaithfulnessを判定し、その結果もリリースゲートに含めます。評価者用の `OPENAI_API_KEY` を安全に設定し、予算を確認してから実行します。

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".ml-labs\venv-rag-eval"
uv sync --extra dev --extra deepeval
uv run ml-labs rag release evaluate --candidate .ml-labs\rag-demo\v1\candidate.json --baseline .ml-labs\rag-demo\v1\candidate.json --deepeval --deepeval-threshold 0.7 --report .ml-labs\rag-demo\v1-semantic-report.json
Remove-Item Env:UV_PROJECT_ENVIRONMENT
```

閾値は0～1です。答えられるケースの選択的評価が不合格なら、通常の検索テストが通ってもゲートは不合格です。評価者自身のトークン・請求額をこのアダプターは計測していないため、推定値を実測として報告しません。

Langfuseは既存プロジェクトの `LANGFUSE_PUBLIC_KEY`、`LANGFUSE_SECRET_KEY` と必要に応じて `LANGFUSE_HOST` を設定し、`--trace`を明示した質疑だけ送信します。

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".ml-labs\venv-rag-trace"
uv sync --extra dev --extra langfuse
uv run ml-labs rag release query --store .ml-labs\rag-demo\store --query "model registry champion" --trace
Remove-Item Env:UV_PROJECT_ENVIRONMENT
```

通常の質疑では外部traceを作りません。送信する場合は質問・検索文書・回答・プロンプトが外部に渡るため、公開合成データだけを使います。

## 10. 既存Azure OpenAIで同じパイプラインを生成モードにする

**ここからは任意の有料経路**です。実行前に既存のリソース、権限と予算を確認します。このコマンドでクラウドリソースやSearchインデックスは作りません。検索は引き続き固定コーパスのローカルTF-IDFです。

```powershell
$azureConfig = Get-Content .ml-labs\rag-demo\inputs\config.json -Raw | ConvertFrom-Json
$azureConfig.provider = "azure_openai"
$azureConfig | Add-Member deployment "<existing-chat-deployment>"
$azureConfig | Add-Member api_version "2024-10-21"
$azureConfig | Add-Member max_tokens 512
$azureConfig | ConvertTo-Json -Depth 10 | Set-Content .ml-labs\rag-demo\azure-config.json -Encoding utf8
$env:AZURE_OPENAI_ENDPOINT = "https://<existing-resource>.openai.azure.com"
$env:UV_PROJECT_ENVIRONMENT = ".ml-labs\venv-rag-azure"
uv sync --extra dev --extra azure
uv run ml-labs rag release create --corpus .ml-labs\rag-demo\inputs\documents.jsonl --config .ml-labs\rag-demo\azure-config.json --prompt .ml-labs\rag-demo\inputs\prompt.json --evaluation .ml-labs\rag-demo\inputs\evaluation.json --output .ml-labs\rag-demo\azure
uv run ml-labs rag release evaluate --candidate .ml-labs\rag-demo\azure\candidate.json --baseline .ml-labs\rag-demo\v1\candidate.json --report .ml-labs\rag-demo\azure-report.json
Remove-Item Env:UV_PROJECT_ENVIRONMENT
```

認証には `DefaultAzureCredential`を使用します。既存デプロイに対する推論権限が必要です。プロバイダーを変えても評価者とサービスの実装は同じです。デプロイ名は設定に含めますが、資格情報は保存しません。

デプロイ名が同じでもクラウド側のモデル実体が変更される可能性があります。実際の応答にはproviderのモデル名とrequest IDも記録しますが、クラウドモデルの不変性までは保証しません。

生成文は抜粋文と同一とは限らないため、`answer_contains`の強すぎる条件では良い言い換えも失敗します。閾値だけを下げるのでなく、期待値をレビューし、[前章のDeepEvalとLangfuse](06-rag-llmops.md)を必要な範囲で接続します。ネイティブPromptfooも同じ候補を直接呼び出せます。これらの外部サービスをこのオフライン手順の検証で実行したとは主張しません。

## 持ち帰る確認事項

- 文書、設定、プロンプト、評価集のどれを変えても候補IDが変わるか。
- 候補と評価報告の取り違え、評価後のファイル変更を拒否するか。
- 存在しない文書のインライン引用は、回答として出力されないか。
- 正解率の平均が同じでも、baselineの正解を失う候補を止められるか。
- API応答から実際に使用したリリースと質疑のtrace IDを追えるか。
- rollbackが以前に有効化した検証済みコピーを参照するか。

今回実際に確認した範囲は、CLIのinit-demo/create/evaluate/activate/query、`doc-feast`を欠く回帰候補のevaluate非ゼロ終了とactivate拒否、ネイティブPromptfooのローカル3/3成功です。ローカルTestClientでも、同じHTTPアプリが有効なリリースを返すこと、回帰候補の拒否後も旧リリースを返すこと、v2有効化とrollbackが次のHTTPリクエストに反映されることを検証しました。

自動テストではさらに、ポインター更新失敗時の旧リリース維持、Azure SDK境界のモック、任意のDeepEval失敗による活性化拒否を検証します。モックはクラウド実行成功の証拠ではありません。Azure生成、DeepEvalの外部評価とLangfuse送信は今回実行していません。

```powershell
uv run python -m pytest tests\test_rag_lifecycle.py -q
```

次の改善は実務用の評価データです。似た単語を含む無回答問題、複数文書が必要な問題、古い文書と新しい文書の競合を追加し、手動評価と自動ゲートが食い違う理由を説明してください。

[基礎: RAGとLLMOps](06-rag-llmops.md) / [Azure実行](07-azure.md)
