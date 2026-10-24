# 07. 로컬 실행 계약을 Azure ML로 옮기기

컴퓨터를 바꿔 실행한다고 데이터와 모델의 관계가 저절로 유지되지는 않습니다. 로컬 경로, 인증, 실행 환경과 결과 저장 위치가 달라집니다. 이번 과정은 같은 Python 구현을 Azure ML의 독립된 작업으로 옮기는 데 초점을 둡니다.

이 문서의 배포와 제출 명령은 사용자가 비용과 권한을 검토한 뒤 실행하는 선택 단계입니다. 자료 작성 과정에서는 Azure 리소스 생성, 작업 제출과 엔드포인트 배포를 실행하지 않았습니다.

## 1. 로컬에서 구성만 만들기

```powershell
uv run ml-labs azure pipeline --output .ml-labs/azure/pipeline.yml --compute azureml:cpu-cluster
az bicep build --file infra/azure/main.bicep --outfile .ml-labs/azure/main.json
```

첫 명령은 파이프라인과 아홉 개의 component YAML을 만듭니다. 각 프로젝트가 `prepare → train/index → evaluate`로 연결됩니다. 두 번째 명령은 Bicep을 ARM JSON으로 컴파일할 뿐 리소스를 생성하지 않습니다.

GPU 없이 실제 YOLO로 바꿀 때는 별도 구성을 만듭니다. 기본 구성의 카드 모델은 색상 규칙 기준선입니다.

```powershell
uv run ml-labs azure pipeline --output .ml-labs/azure/yolo-pipeline.yml --compute azureml:cpu-cluster --cards-backend yolov8
```

YOLO 학습과 평가 단계는 `environment-vision.yml`을 사용합니다. 사전학습 가중치 다운로드와 CPU 학습 시간도 고려해야 합니다.

## 2. 필요한 리소스와 권한 확인하기

`infra/azure/main.bicep`은 Storage, Key Vault, ACR, Application Insights와 Azure ML workspace를 정의합니다. `deployCompute=true`일 때만 최소 노드 0, 최대 노드 2의 CPU 클러스터를 추가합니다. `deployRagServices=true`이면 AI Search와 Azure OpenAI 및 채팅 배포도 추가합니다.

이 구성은 공개 네트워크를 사용하는 학습용 출발점입니다. Private Endpoint, VNet, 조직 정책, 고객 관리 키와 운영용 권한 분리는 구현하지 않았습니다. 학습 구독의 전용 리소스 그룹에서만 검토하세요.

배포 계정에는 리소스 생성 권한과 role assignment 생성 권한이 필요합니다. workspace의 managed identity에는 Blob, Key Vault와 ACR 역할이 부여됩니다. 개발자가 데이터를 읽거나 작업을 제출할 권한은 별도로 확인해야 합니다. 역할 전파에는 시간이 걸릴 수 있습니다.

AI Search에서는 문서를 쓰는 `Search Index Data Contributor`와 인덱스 정의를 관리하는 `Search Service Contributor`가 다른 역할입니다. OpenAI 호출에는 `Cognitive Services OpenAI User`가 필요합니다. `developerPrincipalId`는 사용자 object ID용이며 서비스 principal을 넣으려면 템플릿의 principal type도 바꿔야 합니다.

## 3. 비용이 발생하는 선택 배포

아래 자리표시자를 실제 학습 구독과 지역으로 바꾸고 Azure Portal에서 현재 구독을 확인합니다. 리소스 그룹을 먼저 별도로 준비해야 합니다.

```powershell
az login
az account show
az deployment group what-if --resource-group <lab-resource-group> --template-file infra/azure/main.bicep --parameters namePrefix=<unique-prefix> deployCompute=true
```

`what-if`도 Azure에 접근하는 관리 작업이지만 실제 리소스를 배포하지는 않습니다. 검토가 끝난 뒤에만 다음 명령을 실행합니다.

```powershell
az deployment group create --resource-group <lab-resource-group> --template-file infra/azure/main.bicep --parameters namePrefix=<unique-prefix> deployCompute=true
```

OpenAI 모델의 지역 지원, 버전 수명과 quota는 별도로 확인합니다. 템플릿의 예시 모델 버전이 현재 구독에서 사용 가능하다는 보장은 없습니다. RAG 리소스를 배포할 때는 `chatModelName`, `chatModelVersion`, `developerPrincipalId`도 확인해서 넘깁니다.

Storage, ACR, Search 등의 비용은 학습 작업이 없을 때도 생길 수 있습니다. 클러스터의 최소 노드가 0이라는 것은 전체 환경의 비용이 0이라는 뜻이 아닙니다. 실행 전에 구독 예산과 알림을 설정하세요.

## 4. 기존 workspace 연결하기

루트의 `config.json`을 다음 구조로 작성합니다. 이 파일은 Git에서 제외됩니다.

```json
{
  "subscription_id": "<subscription-id>",
  "resource_group": "<lab-resource-group>",
  "workspace_name": "<workspace-name>"
}
```

`uv run --extra azure`를 사용하면 SDK v2와 MLflow 연동 플러그인이 설치됩니다. CLI의 `--execute`는 리소스 구성을 만드는 옵션이 아니라 이미 렌더링한 작업을 기존 workspace에 제출하는 옵션입니다.

```powershell
uv run --extra azure ml-labs azure pipeline --output .ml-labs/azure/pipeline.yml --execute
```

이 단계부터 기존 compute가 실행되어 비용이 발생합니다. 제출 함수는 workspace의 MLflow URI를 각 작업에 전달합니다. 로컬 SQLite 기록이 Azure에 복사되는 것은 아닙니다.

업로드 때 `.git`은 제외되므로 렌더링한 component command에 코드 커밋이 포함됩니다. 렌더링 후 소스를 수정했다면 커밋하고 YAML을 다시 만드세요. 작업에 기록된 커밋과 실제 업로드 코드가 달라지지 않게 해야 합니다.

## 5. 작업 결과 확인하기

```powershell
az ml job show --name <job-name> --resource-group <lab-resource-group> --workspace-name <workspace-name>
az ml job stream --name <job-name> --resource-group <lab-resource-group> --workspace-name <workspace-name>
```

Azure CLI의 `ml` extension이 없다면 `az extension add --name ml`로 설치합니다. Studio에서 컴포넌트 입력과 출력, 환경 빌드 로그, run의 MLflow 모델과 평가 보고서를 확인합니다.

`Completed`만 보지 말고 다음을 확인합니다.

| 확인 대상 | 봐야 할 근거 |
| --- | --- |
| 데이터 전달 | 학습 입력이 prepare의 `data` 출력인지 |
| 독립 실행 | 다른 작업의 로컬 경로 대신 선언된 output을 쓰는지 |
| 평가 | 학습 후보와 같은 데이터 지문인지, 통과 보고서가 있는지 |
| 추적 | workspace tracking URI, run ID와 모델 아티팩트가 연결되는지 |
| 실패 | 평가 게이트 거절이 작업 실패로 드러나는지 |

현재 파이프라인은 평가까지 연결합니다. 자동 승인, Registry 승격과 운영 엔드포인트 트래픽 변경은 포함하지 않습니다. 책의 KFP 학습을 Azure ML 제출로 대체하는 것도 아닙니다.

## 6. 로컬 검증 모델을 원격 엔드포인트로 내보내기

먼저 로컬에서 평가를 통과한 후보와 보고서를 사용합니다. 클라우드 job에서 받은 candidate를 쓰려면 해당 tracking URI에 접근하는 인증과 모델 아티팩트 권한이 필요합니다.

```powershell
uv run ml-labs azure endpoint --candidate .ml-labs/runs/movies/candidate.json --report .ml-labs/reports/movies.json --output .ml-labs/azure/movie-endpoint
```

이 명령은 모델 파일과 `deploy_endpoint.py`를 내보낼 뿐 원격 엔드포인트를 생성하지 않습니다. 같은 출력 폴더의 기존 모델은 덮어쓰지 않습니다.

생성된 Python을 읽고 workspace 설정, VM SKU와 endpoint 이름을 확인한 뒤 명시적으로 실행합니다.

```powershell
uv run --extra azure python .ml-labs/azure/movie-endpoint/deploy_endpoint.py --endpoint-name <unique-endpoint-name>
```

이 스크립트는 등록 모델을 배포하는 것과 별개로, MLflow 아티팩트를 모델 자산으로 올리고 managed endpoint와 `blue` deployment를 만듭니다. 모델의 MLflow 환경으로 의존성을 복원하는 no-code inference 경로입니다. 비용이 발생하며 instance count는 1입니다.

영화 요청을 만들고 호출합니다.

```powershell
$request = @{ input_data = @{ columns = @("user_id"); data = @(@("u000")) } }
$request | ConvertTo-Json -Depth 8 | Set-Content .ml-labs/movie-request.json
az ml online-endpoint invoke --name <unique-endpoint-name> --request-file .ml-labs/movie-request.json --resource-group <lab-resource-group> --workspace-name <workspace-name>
```

카드 모델은 [base64 이미지 요청](02-cards.md)을 사용하고, 로컬 RAG PyFunc는 `query` 열을 사용합니다. PyFunc를 원격 배포해도 TF-IDF 인용 응답이 OpenAI 생성으로 바뀌지는 않습니다. OpenAI 연결은 별도 RAG provider 경로입니다.

## 7. 정리하기

작업 취소와 엔드포인트 삭제는 서로 다릅니다.

```powershell
az ml job cancel --name <running-job-name> --resource-group <lab-resource-group> --workspace-name <workspace-name>
az ml online-endpoint delete --name <unique-endpoint-name> --resource-group <lab-resource-group> --workspace-name <workspace-name> --yes
```

취소할 작업이 없으면 첫 명령은 필요 없습니다. endpoint 삭제 후 deployment VM이 제거되었는지 확인합니다. Search, OpenAI 배포, ACR과 Storage는 남으므로 사용하지 않으면 학습용으로 생성한 각 리소스를 Portal에서 확인하고 삭제합니다. 기존 업무용 workspace가 들어 있는 리소스 그룹 전체를 삭제하지 마세요.

Key Vault는 purge protection 때문에 삭제 후에도 즉시 이름을 재사용할 수 없습니다. 보존 기간과 구독별 정책도 정리 계획에 포함합니다.

한 줄 정리: 렌더링, Azure 검증, 작업 제출과 엔드포인트 생성은 서로 다른 단계이며 비용과 성공 근거도 따로 확인해야 합니다.

참고: [SDK v2 component pipeline](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-create-component-pipeline-python?view=azureml-api-2), [MLflow endpoint](https://learn.microsoft.com/en-us/azure/machine-learning/how-to-deploy-mlflow-models-online-endpoints?view=azureml-api-2), [Search RBAC](https://learn.microsoft.com/en-us/azure/search/search-security-rbac).

[이전: RAG와 LLMOps](06-rag-llmops.md) / [다음: 책과의 대응](08-coverage.md)
