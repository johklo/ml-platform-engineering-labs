# ML Platform Engineering Labs

모델을 한 번 실행하는 것과, 어떤 데이터로 만든 모델인지 설명하고 다시 배포하는 것은 다른 일입니다. 이 저장소는 『Machine Learning Platform Engineering』에서 다루는 흐름을 세 프로젝트로 실습하는 독립 학습 자료입니다. 책이나 저자의 공식 코드 저장소는 아닙니다.

영화 추천, 합성 신분증 모양 객체 탐지, 문서 검색을 데이터 준비부터 실험 기록, 평가, 등록, 추론까지 연결합니다. 책의 예제를 그대로 복제하지 않고 실행 계약과 실패 실험을 추가했습니다.

## 먼저 실행하기

Python 3.11 또는 3.12, Git, [uv](https://docs.astral.sh/uv/)가 필요합니다. 아래 명령은 PowerShell 기준이며 저장소 루트에서 실행합니다. Linux와 macOS에서는 파일 경로의 구분자를 바꾸면 됩니다.

```powershell
git clone https://github.com/johklo/ml-platform-engineering-labs.git
Set-Location ml-platform-engineering-labs
uv sync --locked --extra dev
uv run --extra dev pytest -q
uv run ml-labs smoke --work .ml-labs/smoke
```

마지막 명령은 작은 합성 데이터로 세 프로젝트의 연결을 점검합니다. `cards`, `movies`, `rag` 평가와 `registered` 모델 URI가 출력됩니다. `.ml-labs/smoke` 안에는 데이터, 후보 모델 명세, 평가 보고서와 SQLite 실험 기록이 남습니다. 합성 데이터에서 통과했다는 결과는 실제 모델 품질의 증거가 아닙니다.

## 학습 순서

| 순서 | 가이드 | 직접 확인할 것 |
| --- | --- | --- |
| 00 | [구조와 준비](docs/00-foundation.md) | 데이터, run, 후보, 평가와 등록 버전의 관계 |
| 01 | [영화 추천](docs/01-movies.md) | 시간 분할, SVD, 인기 기준선, 실제 MovieLens 가져오기 |
| 02 | [객체 탐지](docs/02-cards.md) | 색상 기준선과 실제 YOLO 학습의 차이 |
| 03 | [Feast](docs/03-feast.md) | 과거 시점 피처와 materialization, 온라인 조회 |
| 04 | [파이프라인](docs/04-pipelines.md) | KFP의 아티팩트 전달과 AML의 대응 구조 |
| 05 | [서빙과 관측](docs/05-serving-monitoring.md) | 고정 모델 버전, BentoML, HTTP, Evidently |
| 06 | [RAG와 LLMOps](docs/06-rag-llmops.md) | 검색과 생성의 분리, 평가와 외부 서비스 연결 |
| 07 | [Azure 실행](docs/07-azure.md) | 배포 전 확인, AML 작업, 원격 추론, 비용과 정리 |
| 08 | [책과의 대응 및 검증 범위](docs/08-coverage.md) | 실제 구현과 확장 과제, 실행하지 않은 경로 |
| 09 | [피처, 출시와 복구 통합 실험](docs/09-integrated-operations.md) | 실제 Feast 요청 분기, 순위 기준선 비교, 배치 재처리와 출시 복구 |
| 10 | [RAG 변경 평가와 출시](docs/10-rag-release.md) | 프롬프트와 문서 고정, 회귀 차단, 실제 요청 평가와 이전 버전 복구 |

한 장씩 읽고 명령을 실행한 뒤, 각 장의 실패 실험까지 마치는 순서를 권합니다. 처음부터 모든 선택 의존성을 설치할 필요는 없습니다. 도구 버전의 충돌 때문에 `uv sync --all-extras`는 지원하지 않습니다.

전체 운영 흐름의 로컬 재생은 다음 명령으로 실행합니다. `evidence.json`에 실제 피처 공급 전후의 응답, 재시도와 출시 복구 기록이 남습니다.

```powershell
uv run --locked --extra dev --extra feast --extra monitor ml-labs exercise --work .ml-labs/integrated-01
```

RAG는 `rag release init-demo`에서 시작해 불변 후보 생성, 이전 후보와 평가 비교, 게이트 통과 후 활성화, 실제 API 요청과 rollback까지 이어집니다. [10편](docs/10-rag-release.md)에는 무료 로컬 경로와 실제 런타임을 호출하는 Promptfoo 실행을 함께 정리했습니다. 로컬 인용 경로가 외부 LLM의 생성 품질까지 검증하는 것은 아닙니다.

## 실행 범위

기본 과정은 GPU, Docker, 클라우드 계정 없이 CPU에서 실행합니다. MLflow는 실제 Tracking과 Registry를 사용합니다. 색상 객체 탐지는 학습 모델이 아닌 기준선이며, 로컬 RAG의 기본 응답은 LLM 생성이 아닌 검색 문장 인용입니다. YOLO, Feast, KFP, BentoML, Evidently와 LLMOps 도구는 별도 과정에서 실제 SDK를 사용합니다.

Azure 구성은 리소스를 생성하지 않고 검토할 수 있습니다. 배포나 작업 제출을 명시적으로 실행하면 비용이 발생합니다. 이 자료를 작성하면서 Azure 리소스를 생성하지 않았으며 클라우드 실행 성공을 주장하지 않습니다.

실제 신분증, 책 원문, 비공개 문서, API 키와 MovieLens 데이터를 커밋하지 않습니다. 로컬 API는 인증 없는 학습용이므로 외부에 공개하지 않습니다.

## 참고 자료

- [책과 Manning liveBook](https://livebook.manning.com/book/machine-learning-platform-engineering)
- [저자 코드 모음](https://github.com/practical-mlops)
- [블로그의 책 읽기 지도](https://johklo.github.io/docs/tech/machine-learning-platform-engineering/00-overview/)
- [기존 MLflow와 Azure ML 기초 실습](https://johklo.github.io/docs/tech/mlflow-azure-machine-learning/)

이 저장소의 직접 작성한 코드와 문서는 [MIT License](LICENSE)를 따릅니다. 책, 외부 데이터와 도구에는 각각의 라이선스가 적용됩니다. Ultralytics를 사용할 때는 해당 프로젝트의 AGPL 또는 별도 상용 라이선스를 확인하세요.
