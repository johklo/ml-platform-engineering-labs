# 06. 검색부터 평가·배포까지 같은 RAG를 사용하기

RAG를 자료를 찾아 설명하는 발표자라고 생각해 봅시다. 평가할 때는 좋은 자료를 주고, 실제 서비스에서는 다른 검색기를 사용한다면 점수가 좋아도 운영 품질은 알 수 없습니다. 이 실습에서는 **문서, 검색 설정, 프롬프트와 평가 질문을 하나의 후보로 고정**하고, 그 후보를 검색·응답·평가·배포에서 그대로 사용합니다.

먼저 구분할 것이 있습니다. 기본 구현은 TF-IDF 검색 결과를 인용하는 `extractive_offline`입니다. 외부 호출이나 언어 모델 생성은 없습니다. 생성 실습은 명시적으로 선택하는 `azure_openai`이며, 이미 존재하는 Azure OpenAI 배포를 사용합니다.

## 학습 결과와 실행 경로

이 장을 마치면 다음 질문에 답할 수 있어야 합니다.

- 어떤 문서와 프롬프트로 답했는지 해시와 버전으로 확인할 수 있는가?
- 검색된 문서 목록과 실제 답변에 달린 인용을 구분할 수 있는가?
- 문서에 답이 없는 질문이 평가 집합에 포함되어 있는가?
- 새로운 후보가 기존에 맞히던 질문을 틀리면 활성화를 차단하는가?
- 서비스가 사용하는 릴리스와 평가한 후보가 같은가?

실행 흐름은 다음과 같습니다.

```text
문서 JSONL + 검색/생성 설정 + 버전 프롬프트 + 평가 JSON
    → 불변 후보 스냅샷
    → 같은 RAGRuntime으로 질의와 baseline 비교 평가
    → 해시에 묶인 report.json
    → 게이트 통과 시 로컬 릴리스 저장소 활성화
    → HTTP/CLI에서 같은 릴리스 조회
    → 이상 시 이전 릴리스로 rollback
```

전체 명령과 실패 실험은 [10. RAG 릴리스 실습](10-rag-release.md)에 있습니다. 여기서는 도구가 연결되는 이유와 구현 범위를 먼저 확인합니다.

## 첫 로컬 릴리스

프로젝트 루트에서 실행합니다. 기존 디렉터리를 덮어쓰지 않으므로 다시 시작할 때는 새 경로를 선택합니다.

```powershell
uv sync --extra dev
uv run ml-labs rag release init-demo --output .ml-labs\rag-demo\inputs
uv run ml-labs rag release create --corpus .ml-labs\rag-demo\inputs\documents.jsonl --config .ml-labs\rag-demo\inputs\config.json --prompt .ml-labs\rag-demo\inputs\prompt.json --evaluation .ml-labs\rag-demo\inputs\evaluation.json --output .ml-labs\rag-demo\v1
uv run ml-labs rag release evaluate --candidate .ml-labs\rag-demo\v1\candidate.json --report .ml-labs\rag-demo\v1-report.json
uv run ml-labs rag release activate --store .ml-labs\rag-demo\store --candidate .ml-labs\rag-demo\v1\candidate.json --report .ml-labs\rag-demo\v1-report.json
uv run ml-labs rag release query --store .ml-labs\rag-demo\store --query "model registry champion promotion"
uv run ml-labs rag release query --store .ml-labs\rag-demo\store --query "quasar teleportation warranty"
```

두 번째 질문은 `status: no_grounding`, 빈 `citations`, 고정된 거절 응답을 반환해야 합니다. 단어가 겹치지 않는 예제이므로 이것만으로 업무용 거절 품질을 증명하지는 못합니다. `min_score`를 조정하고, 단어는 겹치지만 답은 없는 질문을 직접 추가해야 합니다.

후보 디렉터리에 다음 파일이 복사됩니다.

| 파일 | 고정하는 내용 |
| --- | --- |
| `corpus.json` | 문서 ID, 제목과 실제 본문 |
| `config.json` | top-k, 최소 점수, n-gram, provider와 생성 설정 |
| `prompt.json` | 로컬 프롬프트 이름·버전·템플릿 |
| `evaluation.json` | 답할 수 있는 질문과 답할 수 없는 질문 |
| `candidate.json` | 위 네 파일의 SHA-256과 후보 ID |

원본 파일을 편집해도 이미 만든 후보는 바뀌지 않습니다. 후보 안의 파일을 수정하면 검증이 실패합니다. 변경하려면 새 후보를 만들어야 합니다.

## 인용 검증과 groundedness를 혼동하지 않기

로컬 모드는 검색한 본문을 `[문서ID]`와 함께 그대로 붙입니다. Azure 모드는 **동일한 TF-IDF 검색 결과**와 고정 프롬프트를 모델에 전달합니다. 다른 Search 인덱스로 우회하지 않습니다.

Azure의 응답은 `answer`, `citations`, `abstained`를 포함하는 JSON이어야 합니다. 런타임은 다음을 확인합니다.

1. 인용 ID가 실제 검색 결과 안에 있는가?
2. 답변의 `[ID]`와 `citations` 배열이 일치하는가?
3. 답변했는데 인용이 없거나, 거절했는데 인용이 달려 있지 않은가?

형식이나 인용이 잘못되면 근거 없는 답변을 그대로 내보내지 않고 거절하며 `citations_valid: false`를 남깁니다. **유효한 ID를 달았다는 사실만으로 그 문서가 답변의 모든 주장을 지지한다는 뜻은 아닙니다.** 의미적 충실성에는 추가 평가와 사람의 확인이 필요합니다.

## Promptfoo: 별도 프롬프트가 아니라 실제 런타임 호출

릴리스용 설정은 `create_release_promptfoo_config(candidate, output)`으로 만듭니다. 네이티브 Python provider가 프로젝트의 `rag_lifecycle.call_api`를 호출하고, 그 함수가 실제 `query_candidate`를 실행합니다. 평가 질문과 assertion도 후보의 평가 집합에서 가져옵니다.

```powershell
uv run python -c "from ml_labs.llmops import create_release_promptfoo_config; print(create_release_promptfoo_config('.ml-labs\rag-demo\v1\candidate.json', '.ml-labs\rag-demo\promptfoo.yaml'))"
# Node.js와 Promptfoo CLI가 없는 경우에만 별도로 설치합니다.
npm install -g promptfoo
$env:PROMPTFOO_DISABLE_TELEMETRY = "1"
uv run ml-labs tools promptfoo run --config .ml-labs\rag-demo\promptfoo.yaml
```

설정은 생성할 때 사용한 Python 실행 파일을 `pythonExecutable`로 기록합니다. 환경을 옮겼다면 설정을 다시 만듭니다. 실행기는 `--no-cache`로 실제 후보를 재실행합니다. 기본 로컬 후보에는 OpenAI 키가 필요하지 않습니다. Azure 후보를 지정하면 명시적으로 선택한 Azure 생성 호출과 비용이 발생할 수 있습니다.

기존 `tools promptfoo init`은 독립적인 OpenAI 프롬프트 연습용 설정입니다. 릴리스 경로와 다르므로 이 실습에서는 위의 **릴리스용 생성 함수**를 사용합니다. Promptfoo 결과 자체가 활성화 권한이 되는 것은 아니며, 활성화에는 프로젝트가 만든 해시 결합 평가 보고서가 필요합니다.

## DeepEval: 선택한 의미 평가 결과를 게이트에 포함하기

기본 게이트는 기대 문서 포함, 답변 문자열 포함, 거절과 인용 검증을 사용합니다. 모델 기반 의미 평가가 필요하다면 Python API의 `evaluate_candidate(..., deepeval=True)`를 사용합니다. 실제 런타임 응답과 검색 문맥을 `LLMTestCase`에 전달하고 Answer Relevancy와 Faithfulness를 평가합니다. 답할 수 있는 사례 중 하나라도 이 선택적 검사에 실패하면 활성화할 수 없습니다.

```powershell
# 유료 평가자 호출입니다. OPENAI_API_KEY는 셸에 안전하게 설정합니다.
$env:UV_PROJECT_ENVIRONMENT = ".ml-labs\venv-rag-eval"
uv sync --extra dev --extra deepeval
uv run python -c "from ml_labs.rag_lifecycle import evaluate_candidate; print(evaluate_candidate('.ml-labs\rag-demo\v1\candidate.json', '.ml-labs\rag-demo\v1-semantic-report.json', deepeval=True, deepeval_threshold=0.7)['passed'])"
Remove-Item Env:UV_PROJECT_ENVIRONMENT
```

평가 점수와 이유는 사례별 `deepeval`에 남습니다. 이 어댑터는 평가자의 토큰이나 청구 금액을 측정하지 않으므로 해당 비용을 숫자로 꾸미지 않습니다. 평가 모델의 판단도 완전한 진실이 아닙니다. 이 안내서의 오프라인 검증에서는 외부 평가자를 호출하지 않았습니다.

## Langfuse: 같은 질의를 명시적으로 추적하기

로컬 프롬프트 버전 저장소는 `register_prompt`가 관리합니다. **Langfuse 네이티브 프롬프트 등록·라벨 배포 기능을 구현했다고 주장하지 않습니다.** 대신 선택한 실제 질의를 Langfuse v2 trace/span으로 보내고, 후보 ID, 문서·설정·프롬프트 해시와 실행 결과를 메타데이터로 기록합니다.

```powershell
# 기존 프로젝트의 LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY,
# 필요 시 LANGFUSE_HOST를 셸에 설정한 뒤 실행합니다.
$env:UV_PROJECT_ENVIRONMENT = ".ml-labs\venv-rag-trace"
uv sync --extra dev --extra langfuse
uv run python -c "from ml_labs.rag_lifecycle import query_release; print(query_release('.ml-labs\rag-demo\store', 'model registry champion', trace=True))"
Remove-Item Env:UV_PROJECT_ENVIRONMENT
```

키가 없으면 요청을 보냈다고 표시하지 않고 오류를 반환합니다. 질문·답변·검색 문서와 프롬프트가 외부 서비스에 전송되므로 공개 합성 자료로만 연습합니다. 일반 질의에서는 Langfuse를 초기화하지 않습니다.

## 측정값과 아직 구현하지 않은 것

- `retrieval_ms`, `generation_ms`, `latency_ms`는 `perf_counter`로 측정합니다. `load_ms`는 스냅샷 검증·TF-IDF 준비, `total_ms`는 전체 함수 경로를 포함합니다.
- 로컬 모드의 모델 호출은 0회입니다. 모델 토크나이저를 실행하지 않으므로 토큰은 `null`, 모델 호출 비용은 0입니다.
- Azure 토큰은 응답의 실제 `usage`에서만 읽습니다. 청구 금액을 알 수 없으므로 `cost_usd`는 `null`입니다. 고정 단가로 가짜 비용을 만들지 않습니다.
- RedisVL의 기존 index/query 연습은 실제 Redis 벡터 검색 어댑터입니다. **이 릴리스 경로에는 의미 기반 답변 캐시를 붙이지 않았습니다.** 캐시를 넣으려면 릴리스·사용자 권한·질의 경계, TTL과 문서 변경 무효화를 먼저 설계해야 합니다.
- 기존 Guardrails 연습은 구조 검사입니다. 새 런타임의 인용 검증과 마찬가지로 완전한 환각·보안 방어가 아닙니다.
- 이 과정은 공개 합성 문서용 단일 호스트 실습입니다. 사용자 인증, 문서별 ACL, 분산 잠금과 서명된 배포 증명은 구현하지 않았습니다.

## 기존 MLflow 검색 기준선도 유지하기

과거 명령을 사용하는 과정은 그대로 실행할 수 있습니다.

```powershell
uv run ml-labs rag prepare --output data\documents
uv run ml-labs rag index --data data\documents --output .ml-labs\runs\rag --top-k 3
uv run ml-labs rag query --candidate .ml-labs\runs\rag\candidate.json --query "What does a feature store do?"
uv run ml-labs rag evaluate --candidate .ml-labs\runs\rag\candidate.json --report .ml-labs\reports\rag.json
```

이 기준선은 MLflow pyfunc 패키징과 네 개의 검색 smoke test를 배우는 경로입니다. 새 릴리스 게이트, 후보 비교와 rollback을 경험하려면 `rag release` 경로를 사용합니다. 기존 `rag query --provider azure`는 Azure AI Search의 키워드 검색을 별도로 사용하는 이전 실습입니다. 그것을 새 릴리스의 고정 코퍼스 평가 결과로 간주하면 안 됩니다.

한 줄 정리: 도구별 성공 화면보다 중요한 것은 **평가한 문서·설정·프롬프트로 실제 서비스가 응답한다는 연결**입니다.

[이전: 서빙과 관측](05-serving-monitoring.md) / [다음: Azure 실행](07-azure.md) / [심화: RAG 릴리스](10-rag-release.md)
