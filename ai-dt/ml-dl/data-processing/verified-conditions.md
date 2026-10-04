---
tags: [ml, data-processing, documentation-review]
reviewed_on: 2026-10-04
review_status: partial
document_type: learning
---

# 데이터 처리의 근거와 적용 조건

## 실행 순서와 대상

대부분의 코드 블록은 같은 문서 앞의 import·변수·클래스를 사용하는 **부분 예제**다. `df`, `X_train`, `scaler` 등이 이미 존재해야 한다. API 주소·파일명은 예시이며 실제 회사 자료나 서비스가 아니다. Titanic 예제의 `fetch_openml`과 `sns.load_dataset`은 네트워크 또는 cache가 필요하다. 로컬 합성 데이터 확인을 실제 Titanic 다운로드·API 성공으로 해석하지 않는다.

최종 test는 보류한 채 학습 데이터로 EDA·변환 선택·튜닝을 수행한다. EDA로 test 전체의 분포와 타겟 관계를 보고 피처를 결정하면 간접 누수가 생길 수 있다. 장비·대상·시계열이 반복되면 랜덤 분할 대신 실제 적용 단위를 반영한 그룹/시간 분할이 필요하다. **Pipeline은 분할 전 누수나 타겟이 포함된 입력을 자동 감지하지 않는다.**

## 파일 로딩과 결측

CSV는 타입 스키마가 없어 로더가 추론한다. `001` 같은 식별자, 날짜·timezone·통화·결측 표시는 명시적인 스키마로 검증한다. 오류 없이 decode됐다는 것만으로 올바른 인코딩을 판별할 수 없다. 후보 재시도는 출처 인코딩 확인의 대체가 아니다.

pandas 3은 기본 문자열 dtype이 `str`로 바뀌었다. pandas 2/3 공통 예제는 `object`와 `string`을 함께 선택한다. `object`에는 숫자·목록도 들어갈 수 있어 문자열 정리 전에 내용 타입을 확인한다. 빈 데이터에서 비율 분모는 따로 처리하고 미확인 결측을 관찰 단계에서 0·`unknown`으로 자동 바꾸지 않는다. 실제 대치는 도메인 정책과 학습 fold의 통계로 결정한다. category에 새로운 대치값을 넣을 때는 먼저 그 category를 등록해야 한다.

Excel 예제는 원본 첫 행이 제목, 두 번째 행이 실제 헤더인 경우다. `skiprows=[0]` 뒤에는 `header=0`을 사용한다. sheet·엔진·저장된 cell 타입을 확인하며 화면 표시의 선행 0이 원본 숫자 셀에서 복구된다고 보장하지 않는다. Excel 폴더에 파일이 없으면 `pd.concat([])`이 실패하므로 호출자가 파일 존재를 확인해야 한다.

Parquet은 스키마·컬럼 선택에 유리하지만 모든 데이터에서 속도·압축 우위가 보장되지는 않는다. pandas 예제의 pyarrow 엔진·추가 의존성을 설치해야 하며 날짜 filters는 실제 저장 타입에 맞춘다. Polars의 CSV 타입 지정은 `schema_overrides`를 사용한다(`dtypes`는 0.20.31에서 이름 변경). lazy `collect()`가 곧 모든 입력의 제한된 메모리 스트리밍 보장은 아니다. JSON Lines 저장과 API JSON 응답 계약도 구분한다.

## EDA 결과 해석

상관계수는 인과관계나 비선형 관계 전체를 나타내지 않는다. IQR·z-score의 기준 밖 값은 점검 후보이며 센서 오류·정상 희귀 상태·실제 고장을 도메인으로 판별해야 한다. 결측은 검출되지 않은 정상으로 바꾸지 않고 nullable mask로 보존했다. constant 열은 z-score 분모가 0이므로 유효 값의 이상치 mask를 False로 처리한다. 이 정책도 이상치가 없다는 장비 진단을 뜻하지 않는다.

행 정규화 crosstab의 margin은 합계 **행**으로 표현되므로 세 개의 열 이름을 강제로 넣지 않는다. 타겟 값 0/1의 의미는 Titanic 예제에만 해당한다. 수치 열이 없을 때 plot 함수의 행 수를 0으로 만들지 않는다. 빈 입력의 run_full_eda는 status=empty와 빈 결과를 반환한다. `run_full_eda`는 컬럼 종류에 따라 `outliers`·`correlation` 키를 만들지 않을 수 있으므로 소비자는 키 존재를 확인한다. 결측 0행 비율은 관측 결과가 아니라 빈 표 처리다.

seaborn의 dataset은 예제용 다운로드 자료다. 한글 폰트는 OS 이름만으로 설치가 보장되지 않는다. headless 실행은 계산·plot 생성의 확인이며 실제 한글 글리프·사용자 화면·missingno 설치와 모든 시각화 품질을 확인한 것은 아니다. 경고를 전부 숨기면 폐기 API·dtype·폰트 문제를 놓칠 수 있어 기본 예제의 일괄 경고 억제를 제거했다.

## 변환과 Pipeline

StandardScaler는 학습 평균·분산으로 스케일을 조정하며 정규분포로 만들지 않는다. MinMaxScaler는 기본 설정에서 새 값이 학습 범위를 벗어나면 [0,1] 밖을 반환할 수 있다. RobustScaler는 중앙값·IQR을 사용하지만 이상치를 자동 제거하지 않는다. TF-IDF가 CountVectorizer보다 항상 좋다는 보장은 없다. 모델·분할·평가 과제에 맞게 비교한다.

순서 없는 고카디널리티를 ordinal로 바꾸면 인위적인 순서가 생긴다. AutoPreprocessor의 임계치 fallback은 이전 예제를 보존한 교육용 선택이며 최적 정책이 아니다. OneHot의 `handle_unknown="ignore"`는 미등록 값을 0 벡터로 표현해 변환 실패를 줄이며 모델 품질을 보장하지 않는다. 알려진 범주 drop 설정과 결합하면 다른 의미가 혼동될 수 있다. Ordinal 미등록 -1도 입력 의미를 따로 관리해야 한다.

AutoPreprocessor는 이름·순서가 일정한 비어 있지 않은 DataFrame을 받는다. 지원하지 않는 날짜 등 타입은 조용히 버리지 않고 명시적인 변환을 요구한다. 문자열·nullable boolean의 결측을 imputer용 object/np.nan으로 맞추고 `keep_empty_features=True`로 학습 중 전부 결측인 열의 출력 차원을 보존한다. 이 열의 기본 대치값·업무 의미를 검증해야 한다. 혼합 객체, list·dict cell, infinity·날짜 변환, 대형 dense one-hot 메모리는 이 템플릿의 완전 지원 범위가 아니다.

DatetimeFeatureExtractor는 constructor 매개변수를 그대로 보존해 sklearn clone과 맞추고, 실제 transform 순서의 피처 이름을 반환한다. 시간대·결측 날짜·기준 시점은 사용자가 정의해야 한다. `Timestamp.now()` 기반 경과일은 재실행 시 바뀌므로 실제 학습/서빙에는 기준 시점을 고정한다. qcut의 중복 경계, 로그의 `x > -1` 조건, 비율 분모 0도 입력 계약에 포함한다. 교재에서 전체 df에 fit한 단독 변환은 동작 설명이며 평가에는 train에서만 fit한다.

Pipeline 마지막 단계는 fit 및 사용할 작업의 predict/transform 등을 제공해야 한다. `set_output(transform="pandas")`는 변환기의 피처 이름 계약과 dense 출력이 필요하다. 같은 문서에서 `pipe` 변수가 여러 종류로 재할당되므로 `named_steps["preprocessor"]` 예제는 그 이름의 단계가 있는 파이프라인에만 사용한다.

joblib은 pickle 기반이라 로드 시 코드가 실행될 수 있다. 신뢰한 자체 산출물만 읽고 패키지 판본·입력 스키마·custom class import 경로를 기록한다. 서로 다른 sklearn 판본 간 로딩은 지원 계약이 아니다. 로컬 own-artifact roundtrip은 다른 OS·배포 환경 호환성 인증이 아니다.

## 공식 출처 — 2026-10-04 확인

| 자료 | 판본·확인한 범위 |
|---|---|
| [pandas 문자열 전환](https://pandas.pydata.org/docs/user_guide/migration-3-strings.html) | 확인 화면 3.0.6 문서; default dtype·2/3 공통 선택. 로컬 pandas는 3.0.3. |
| [pandas crosstab](https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html) | 3.0.6 문서; 행 정규화·margin. |
| [pandas read_excel](https://pandas.pydata.org/docs/reference/api/pandas.read_excel.html) | 3.0.6 문서; header·skiprows·dtype. |
| [pandas read_parquet](https://pandas.pydata.org/docs/reference/api/pandas.read_parquet.html) | 3.0.6 문서; engine·columns·filters. |
| [Polars read_csv](https://docs.pola.rs/api/python/stable/reference/api/polars.read_csv.html) | stable 문서 확인일 화면; schema_overrides 이름 변경. 로컬 1.44.2. |
| [sklearn 누수·재현 조건](https://scikit-learn.org/stable/common_pitfalls.html) | 1.9.1 문서; Pipeline·CV·randomness. |
| [Pipeline 계약](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) | 1.9.1 문서; 중간·마지막 단계. |
| [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) | 1.9.1 문서; 결측 sentinel·keep_empty_features. |
| [sklearn estimator 개발 계약](https://scikit-learn.org/stable/developers/develop.html) | 1.9.1 문서; constructor·clone·피처 이름. |
| [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) | 1.9.1 문서; 학습 통계·상수 열. |
| [MinMaxScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) | 1.9.1 문서; 범위·clip 조건. |
| [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html) | 1.9.1 문서; sparse_output·미등록 값. |
| [sklearn 모델 저장](https://scikit-learn.org/stable/model_persistence.html) | 1.9.1 문서; 신뢰 출처·판본 조건. |
| [Matplotlib subplots](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html) | 확인 화면 3.11.2 문서; squeeze와 axes 형태. |
| [seaborn load_dataset](https://seaborn.pydata.org/generated/seaborn.load_dataset.html) | 0.13.2 문서; 온라인 dataset/cache. |

속도 배수·실제 회사 파일·모델 일반화·모든 optional API·외부 예제 다운로드는 미확인이다. 이 문서는 확인한 API 계약과 로컬 검증 범위를 기록하며 최신 버전의 일괄 인증을 뜻하지 않는다.
