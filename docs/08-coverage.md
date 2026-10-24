# 08. 책과 실습의 대응 및 검증 범위

이 저장소는 책의 전체 예제를 그대로 실행하는 공식 배포판이 아닙니다. 동일한 문제를 독립 구현하고, 데이터 출처와 평가 계약을 추가한 학습 자료입니다. 어떤 도구를 실제로 실행하는지와 어떤 기능은 추가 과제인지 구분합니다.

## 책과 연결해서 보기

| 책의 구간 | 저장소에서 실행하는 부분 | 아직 포함하지 않은 부분 |
| --- | --- | --- |
| 1~3장 기반 환경 | Python 환경 고정, CI, 컨테이너화 경로 | Kubernetes 설치, Helm 플랫폼 설치, GitLab CI와 Argo CD 운영 |
| 4장 실험과 피처 | MLflow Tracking과 Registry, Feast 시점 조회와 실제 추천 API 라우팅, 누락과 최신성 정책 | 다중 테넌트 권한, 피처 소유권과 schema migration |
| 5장 오케스트레이션 | 실제 KFP SDK 컴파일, 프로젝트 아티팩트 전달, Azure ML component 구성 | 실행 중인 KFP 클러스터 제공, 운영 스케줄과 장애 대응 자동화 |
| 6장 프로덕션화 | BentoML 서비스 생성, 패키징 명령, Evidently 보고서 | 자동 rollout, autoscaling과 운영 drift 알림 체계 |
| 7장 준비 | 이미지와 YOLO 라벨, 평점 시간 분할, MovieLens 가져오기 | 실제 신분증 수집, 데이터 라벨링 운영과 동의 관리 |
| 8장 탐지 학습 | 색상 기준선과 선택 YOLO CPU 학습, 검증 후 테스트 | 표준 mAP 기반 출시 정책, GPU 분산 학습 |
| 9장 추천 학습 | SVD, 인기 기준선과 검증 Recall/NDCG 비교, 검증 후보 선택, cold start, holdout 평가 | 책의 모든 추천 알고리즘 재현, 대규모 희소 학습과 ANN |
| 10장 추론 | HTTP 추론, BentoML, MLflow 모델 내보내기, AML endpoint 스크립트 | KServe, 부하에 따른 무중단 트래픽 전환 |
| 11장 관측 | Prometheus 형식 계측과 Evidently drift | Grafana, Loki, Alertmanager 배포, Eigen-CAM과 EMF 설명가능성 |
| 12장 RAG | TF-IDF 로컬 인용, Azure Search 키워드 검색과 OpenAI 생성 연결 | 책의 Gemini, FAISS, LangChain, Chainlit 구성 재현 |
| 13장 LLMOps | 불변 프롬프트와 문서 스냅샷, 변경 전후 회귀 평가, 실제 RAG를 호출하는 Promptfoo, 평가 게이트와 출시 및 복구, 선택 DeepEval와 Langfuse 연동 | 의미 캐시, 문서별 권한, Langfuse의 운영 프롬프트 승인, 비용 강제 제한과 종합 안전성 평가 |

Azure ML은 KFP와 비슷한 책임을 맡을 수 있지만 같은 도구는 아닙니다. Azure OpenAI 역시 책의 Gemini 코드를 실행한 것이 아닙니다. 대응되는 책임을 비교하는 별도 경로입니다.

## 자동 검사의 의미

기본 CI는 데이터 계약, 준비 이후 변경 감지, 후보와 평가의 연결, 모델 등록과 고정 버전 재로딩, base64 이미지 입력, HTTP 추론과 작은 검색 회귀를 검사합니다. 세 프로젝트를 함께 실행하는 smoke도 수행합니다.

통합 운영 실험은 실제 Feast 값 공급 전후의 HTTP 응답, 피처 수에 따른 추천 경로, 오래되거나 누락된 값의 거절을 확인합니다. 배치 저장 실패 후 동일 업무 키로 재시도했을 때 결과가 중복되지 않는지, 출시 전환 실패에서 기존 포인터가 유지되고 이전 버전으로 복구되는지도 검사합니다. Evidently 보고서와 결과 근거를 CI artifact로 남깁니다.

RAG는 프롬프트, 문서, 검색 설정과 평가 집합을 고정한 후보에 대해 정상 질문, 근거 없는 질문과 잘못된 인용을 점검합니다. 이전 후보보다 품질이 떨어지면 출시를 차단하며 실제 API가 active release와 rollback을 따르는지 검사합니다. 로컬 결정적 검사를 유료 생성 모델의 품질 검증과 혼동하지 않습니다.

선택 도구의 테스트는 해당 extra가 설치되었을 때 실행합니다. 기본 환경에서 `skipped`로 표시된 검사는 통과한 검사가 아닙니다.

```powershell
uv run --extra dev pytest -q
uv run --extra dev --extra feast pytest -q tests/test_optional_integrations.py -k native_feast
uv run --extra dev --extra pipeline pytest -q tests/test_optional_integrations.py -k native_kfp
uv run --extra dev --extra monitor pytest -q tests/test_optional_integrations.py -k native_evidently
uv run --extra dev --extra azure pytest -q tests/test_cli_and_azure.py
```

KFP 컴파일 검사는 클러스터 실행 성공과 다릅니다. Azure SDK가 YAML을 읽고 검증하는 검사도 구독의 지역, quota, 네트워크와 역할을 검증하지 않습니다. Bicep 로컬 컴파일 역시 실제 배포 성공을 보증하지 않습니다.

외부 호출이 필요한 OpenAI, Langfuse와 Promptfoo의 유료 모델 요청은 기본 CI에서 실행하지 않습니다. RedisVL의 오프라인 테스트는 스키마와 쿼리 구성을 확인하며 실제 Redis 서버의 검색 성공을 증명하지 않습니다. Docker 엔진, KFP 클러스터와 Azure 리소스가 없는 환경에서 그 경로의 성공을 주장하지 않습니다.

## 각 과정을 마친 기준

명령이 종료된 것보다 근거를 설명할 수 있는지를 확인합니다.

| 과정 | 스스로 설명하고 보여 줄 것 |
| --- | --- |
| 영화 추천 | 시간 누수를 차단한 분할, 기준선 비교, 관련 영화 정의와 cold start 응답 |
| 카드 탐지 | 탐지와 OCR의 차이, 기준선의 한계, 양성 IoU와 음성 오검출 |
| Feast | 조회 기준 시점, materialization 전후의 차이, 누락 피처를 정상값과 구분 |
| 파이프라인 | 다른 컨테이너로 전달되는 파일, 실패한 단계와 다음 단계의 관계 |
| 서빙 | 구체 모델 버전, 요청 스키마, 잘못된 입력과 지연 지표 |
| 드리프트 | reference/current의 차이와 성능 저하로 단정하지 않는 이유 |
| RAG | 검색 누락과 생성 오류의 분리, 근거 없는 질문의 응답, 평가의 한계 |
| Azure | 리소스 구성과 job 제출의 차이, 비용을 발생시키는 단계와 삭제 대상 |

## 다음 확장 과제

현재 도구를 설치한 상태에서 아래 과제를 하나씩 추가하면 안내된 경로를 벗어나 설계하는 연습이 됩니다.

1. 이미 구현된 인기 기준선 Recall@5와 NDCG@5 비교를 활동량, 장르별 세그먼트로 확장하고 업무 목표에 맞는 출시 조건을 정의합니다.
2. 탐지에 회전, 조명, 가림 조건을 추가하고 데이터 버전을 나누어 성능 저하를 재현합니다.
3. 단일 호스트에서 확인한 결과 저장 실패와 멱등 재처리를 클러스터 및 object store로 확장하고, 저장소별 조건부 쓰기와 동시성 정책을 구현합니다.
4. 운영 입력의 분포가 바뀌어도 성능은 유지되는 사례를 만들고 drift만으로 재학습하지 않는 판단을 기록합니다.
5. RAG 평가에 정답 근거가 없는 질문, 비슷하지만 다른 문서, 권한이 다른 사용자를 추가합니다.
6. 이미 구현된 RAG 출시 단위와 회귀 평가를 문서별 권한, 실제 모델의 답변 품질, 조직 승인과 점진 배포로 확장합니다.

이 과제들은 현재 구현했다고 주장하는 기능이 아니라, 기본 과정 이후 직접 완성할 부분입니다.

[이전: Azure 실행](07-azure.md) / [전체 과정](../README.md)
