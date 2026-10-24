# 06. 검색과 생성을 나누어 RAG 검증하기

RAG를 자료를 찾아 답하는 발표자라고 생각해 봅시다. 틀린 자료를 찾아 놓고 발표자의 말솜씨만 평가하면 문제를 고치기 어렵습니다. 검색이 필요한 근거를 가져왔는지와 답변이 그 근거에 맞는지는 나누어 평가해야 합니다.

## 외부 호출 없는 검색 기준선

```powershell
uv run ml-labs rag prepare --output data/documents
uv run ml-labs rag index --data data/documents --output .ml-labs/runs/rag --top-k 3
uv run ml-labs rag query --candidate .ml-labs/runs/rag/candidate.json --query "What does a feature store do?"
uv run ml-labs rag evaluate --candidate .ml-labs/runs/rag/candidate.json --report .ml-labs/reports/rag.json
uv run ml-labs register --candidate .ml-labs/runs/rag/candidate.json --report .ml-labs/reports/rag.json --name platform-document-retriever
```

기본 문서는 직접 작성한 짧은 영문 합성 자료입니다. TF-IDF로 문장과 질문의 단어를 벡터화하고 코사인 유사도가 높은 문서를 찾습니다. 응답의 `answer_mode`는 `extractive_offline`입니다. LLM이 새 문장을 만든 것이 아니라 찾은 문장을 문서 ID와 함께 인용합니다.

미리 정한 네 질문에서 기대한 문서가 상위 세 개에 있는지 검사합니다. 이 작은 평가 집합은 검색 연결의 회귀 점검용이지 업무 품질 인증이 아닙니다. `top-k`를 바꾸면 별도 후보로 기록하고 같은 질문으로 비교하세요. 현재 보고서 필드명은 기본값을 따라 `retrieval_recall_at_3`이므로 k를 바꿨다면 후보 설정도 함께 읽어야 합니다.

없는 질문을 넣어 보세요.

```powershell
uv run ml-labs rag query --candidate .ml-labs/runs/rag/candidate.json --query "zxqv987654"
```

공통 단어가 전혀 없으면 근거가 없다는 응답과 빈 인용 목록이 나와야 합니다. 단어가 조금 겹친다는 이유로 엉뚱한 자료가 검색될 수 있으므로 이를 충분한 거절 정책이라고 생각하면 안 됩니다.

## 기존 Azure 서비스로 생성 연결하기

별도의 [Azure 과정](07-azure.md)에서 서비스, 권한과 비용을 확인한 뒤 실행합니다. 이 명령은 이미 존재하는 Search 서비스에 인덱스를 만들고 문서를 업로드합니다.

```powershell
$env:AZURE_SEARCH_ENDPOINT = "https://<search-name>.search.windows.net"
$env:AZURE_SEARCH_INDEX = "ml-platform-public-demo"
$env:AZURE_OPENAI_ENDPOINT = "https://<openai-name>.openai.azure.com"
$env:AZURE_OPENAI_DEPLOYMENT = "<chat-deployment-name>"
uv run --extra azure ml-labs rag azure-index --data data/documents
uv run --extra azure ml-labs rag query --provider azure --query "What does a feature store do?"
```

인증은 `DefaultAzureCredential`과 Microsoft Entra ID를 사용합니다. 키를 코드에 넣지 않습니다. 현재 Search 경로는 키워드 검색이고, OpenAI는 가져온 문맥으로 답변을 생성합니다. 벡터 임베딩 검색이나 하이브리드 검색을 구현했다고 볼 수는 없습니다.

현재 인용 목록은 검색에 전달한 문서 ID 목록이며 생성 문장의 사실성을 자동 보증하지 않습니다. 공개 합성 문서 전용 데모이고 문서별 ACL은 구현하지 않았습니다. 사내 비공개 문서를 넣기 전에 사용자 인증, 검색 단계의 권한 필터, 인용 검증과 기록 보존을 추가해야 합니다.

## Guardrails로 형식 검사하기

```powershell
uv run --extra guardrails ml-labs tools guardrails --candidate .ml-labs/runs/rag/candidate.json --query "What is MLflow?"
```

Guardrails AI의 실제 Pydantic guard로 응답의 구조와 비어 있지 않은 답변을 검사합니다. 이 단계는 응답 스키마 검사입니다. 환각, 개인정보, 프롬프트 주입을 모두 막는 안전 시스템을 구현한 것은 아닙니다.

## DeepEval로 의미 평가하기

평가자도 모델 API를 호출하므로 비용이 발생합니다. 공개 합성 자료만 사용하고 `OPENAI_API_KEY`는 현재 세션에 안전한 방식으로 설정합니다.

```powershell
uv run --extra deepeval ml-labs tools deepeval --candidate .ml-labs/runs/rag/candidate.json --query "How are model versions managed?" --threshold 0.7
```

현재 어댑터는 로컬 인용 응답의 Answer Relevancy를 평가합니다. 생성 응답의 groundedness를 모두 검사하는 것이 아닙니다. 평가자 모델, 라이브러리 버전과 판단 이유를 남기고 사람의 판정과 비교해야 합니다. 자동 실행 결과를 Registry 게이트와 연결하는 작업은 확장 과제입니다.

## Promptfoo로 두 종류의 검사 비교하기

Node.js와 Promptfoo CLI가 별도로 필요합니다.

```powershell
npm install -g promptfoo
uv run ml-labs tools promptfoo init --output integrations/promptfoo/promptfooconfig.yaml
uv run ml-labs tools promptfoo run --config integrations/promptfoo/promptfooconfig.yaml
```

생성된 YAML에는 제공 문맥으로 답해야 하는 질문과 문맥에 답이 없는 질문이 있습니다. 단어 포함 검사와 LLM 평가 기준의 차이를 확인합니다. 이 설정은 별도로 OpenAI 모델에 요청하며 앞에서 만든 전체 RAG API를 호출하는 구성은 아닙니다. 회귀 시험을 확장할 때는 실제 API provider, 고정 질문 집합과 근거 문서를 연결해야 합니다.

## Langfuse로 검색 추적하기

기존 Langfuse 프로젝트의 `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, 필요 시 `LANGFUSE_HOST`를 설정한 뒤 실행합니다.

```powershell
uv run --extra langfuse ml-labs tools langfuse --candidate .ml-labs/runs/rag/candidate.json --query "What does the model registry do?"
```

검색 작업이 span으로 기록되는지 확인합니다. 이 어댑터는 검색 추적이며 프롬프트 저장소와 배포 라벨 관리까지 구현하지 않았습니다. 외부 서비스에 질문과 응답이 전송되므로 실제 개인정보나 사내 문서를 테스트에 넣지 않습니다.

## RedisVL로 검색 저장소 바꾸기

벡터 검색을 지원하는 Redis가 별도로 필요합니다. 기본 Redis 서버에 검색 모듈이 없으면 동작하지 않을 수 있습니다.

```powershell
uv run --extra redisvl ml-labs tools redisvl index --data data/documents --index ml-labs-documents-v1 --state .ml-labs/redisvl/state.joblib
uv run --extra redisvl ml-labs tools redisvl query --query "model registry" --state .ml-labs/redisvl/state.joblib
```

로컬 TF-IDF 벡터를 실제 RedisVL 인덱스에 저장하고 조회합니다. 기존 인덱스를 덮어쓰지 않으므로 다시 만들 때는 새 이름을 사용합니다. 이 기능은 벡터 검색이지 책의 의미 기반 답변 캐시를 완성한 것은 아닙니다. 캐시는 사용자 권한, 문서 버전과 프롬프트 버전까지 키의 경계에 포함해야 합니다.

한 줄 정리: 검색, 생성, 형식 검사와 모델 평가를 나누고, 각 성공이 보장하지 않는 범위를 확인합니다.

[이전: 서빙과 관측](05-serving-monitoring.md) / [다음: Azure 실행](07-azure.md)
