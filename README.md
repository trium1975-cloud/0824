# 💄 Sephora E-Commerce Pricing Strategy & AI Analytics Service

> **세포라(Sephora) 이커머스 데이터를 활용한 가격 티어별 고객 만족도 통계 검정, 바이럴 반응 예측 머신러닝 파이프라인 및 인터랙티브 웹 서비스 프로토타입**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![SciPy](https://img.shields.io/badge/SciPy-Statistics-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)](https://scipy.org/)
[![Scikit--Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Modeling-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Gradio](https://img.shields.io/badge/Gradio-Web%20Service-FF7C00?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-Notebook-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](./sephora_data_analysis.ipynb)

---

## 📑 Table of Contents
1. [Project Overview & System Architecture](#1-project-overview--system-architecture)
2. [Data Engineering & Preprocessing Pipeline](#2-data-engineering--preprocessing-pipeline)
3. [Deep-Dive Exploratory Data Analysis (EDA)](#3-deep-dive-exploratory-data-analysis-eda)
4. [Statistical Hypothesis Testing (One-way ANOVA)](#4-statistical-hypothesis-testing-one-way-anova)
5. [Predictive Machine Learning Modeling](#5-predictive-machine-learning-modeling)
6. [Interactive Gradio Web Service Interface](#6-interactive-gradio-web-service-interface)
7. [Data-Driven Business Strategies](#7-data-driven-business-strategies)
8. [Engineering Retrospective & Troubleshooting](#8-engineering-retrospective--troubleshooting)

---

## 1. Project Overview & System Architecture

### 1.1 비즈니스 문제 정의 (Business Problem)
뷰티 이커머스 플랫폼에서 **"고가 프리미엄 제품일수록 고객 만족도(평점)가 비례하여 상승하는가?"**는 신규 브랜드 소싱, 가격 책정(Pricing), 마케팅 프로모션 예산 배분의 핵심 기준입니다.
* **분석 배경**: 무조건적인 고가 럭셔리 라인업 확장이 실제 고객 리텐션과 높은 평점으로 직결되는지 데이터 기반의 실증적 검증이 요구됨.
* **핵심 가설 1 (만족도 가설)**: 판매 가격이 높을수록 브랜드 신뢰도와 원료 품질 효과로 인해 고객 만족도(Rating)가 선형적으로 유의미하게 높을 것이다.
* **핵심 가설 2 (바이럴 가설)**: 플랫폼 내 등록 상품 수(SKU)가 많은 대중적 카테고리가 고객의 바이럴 반응(Love/위시리스트 수)에서도 동일하게 높은 점유율을 가질 것이다.

### 1.2 엔드투엔드 파이프라인 아키텍처 (Pipeline Flowchart)

```mermaid
flowchart TD
    A[Sephora Raw Dataset: 9,168건] --> B[Data Cleansing: 0.0점 노이즈 398건 식별 및 제거]
    B --> C[Feature Engineering: 4-Tier 비즈니스 가격 구간화]
    C --> D[EDA: 카테고리별 상관관계 및 바이럴 비대칭성 분석]
    C --> E[Statistical Testing: One-way ANOVA F=42.95, p<0.001]
    C --> F[Machine Learning: 하트수 및 평점 예측 Random Forest/XGBoost]
    E & F --> G[Interactive Web App: Gradio 기반 실시간 MD 시뮬레이터 프로토타입]
    G --> H[Actionable Business Strategy: $26~$50 소싱 Sweet Spot 도출]

## 2. Data Engineering & Preprocessing Pipeline

### 2.1 결측치 식별 및 0점 노이즈 정제 (Data Cleansing)
* **원천 데이터 규모**: Sephora Website Dataset 총 9,168건
* **결측/이상치 정의**: 입점 직후 평가가 이루어지지 않은 신제품의 평점 결측치(`rating == 0.0`) 및 가격 오류치(`price <= 0`) 식별
* **정제 결과**: 0점 노이즈 398건을 제거하여 통계적 왜곡을 배제한 **8,770건의 유효 분석 데이터셋** 확보

### 2.2 피처 엔지니어링 (Feature Engineering)
* **4-Tier 비즈니스 가격 구간화**:
  * `Budget`: $25 미만 (가성비 대중 상품군)
  * `Mid-tier`: $25 이상 ~$50 미만 (주력 소비 볼륨 상품군)
  * `Premium`: $50 이상 ~$100 미만 (고기능성 프리미엄 상품군)
  * `Luxury`: $100 이상 (초고가 럭셔리 라인업)
* **이커머스 리뷰 전환율 기반 파생변수 생성**:
  * $\text{Estimated Volume} = \text{number\_of\_reviews} \times 30$ (주문 30건당 리뷰 1건 작성 기준)
  * $\text{Estimated Revenue} = \text{Estimated Volume} \times \text{price}$
* **조기 단종/퇴출 위험군(Churn Risk) 라벨링**:
  * 평점 3.8점 이하 AND 고객 관심도(Love) 하위 25% 이하 조건을 동시 충족하는 비인기·품질 경고 상품을 `Churn Risk = 1`로 이진 분류

---

## 3. Deep-Dive Exploratory Data Analysis (EDA)

### 3.1 카테고리별 편중 및 바이럴 비대칭성
* 상위 10개 핵심 카테고리(Perfume, Moisturizers, Face Serums 등)가 전체 상품 SKU의 60% 이상을 점유
* **바이럴 관심도(Love)의 멱법칙(Pareto) 분포**: 상위 20%의 바이럴 히트 상품이 플랫폼 전체 관심도 하트의 78% 이상을 독점하는 비대칭 구조 확인

### 3.2 다변량 상관관계 매트릭스 (Multivariate Correlation)
* **가격 vs 평점**: 상관계수 $r \approx 0.05$ 수준으로, 고가 상품일수록 만족도가 높을 것이라는 직관적 가설과 달리 선형적 상관관계가 매우 미미함
* **리뷰 수 vs 관심도(Love)**: 상관계수 $r > 0.65$의 강한 양의 상관성을 보여, 고객 신뢰 형성에는 가격보다 소셜 증거(리뷰 수)가 핵심 동인으로 작용함을 규명

---

## 4. Statistical Hypothesis Testing (One-way ANOVA)

### 4.1 가설 설정 및 통계 검정 설계
* **귀무가설($H_0$)**: 4개 가격 티어(Budget, Mid-tier, Premium, Luxury) 간 고객 평점(만족도)의 평균 차이는 없다.
* **대립가설($H_1$)**: 적어도 한 개 이상의 가격 티어 간 고객 평점 평균에 유의미한 차이가 존재한다.

### 4.2 검정 통계량 분석 결과
* **일원분산분석(One-way ANOVA) 결과**:
  * $F\text{-statistic} = 42.95$
  * $p\text{-value} = 1.34 \times 10^{-27}\ (p < 0.001)$
  * 유의수준 1% 하에서 귀무가설을 강력하게 기각, 가격 구간별 만족도 차이가 실재함을 입증
* **사후검정 (Tukey HSD Post-Hoc Analysis)**:
  * Budget 구간과 Luxury 구간 간 평점 차이는 통계적으로 유의미하나, 실질적 평점 차이는 0.15점 내외에 불과함
  * $25~$50 선의 Mid-tier 구간이 가격 대비 품질 불만족 리스크가 가장 낮고 평점이 고르게 상위권에 수렴하는 '만족도 스위트 스팟'임을 확인

---

## 5. Predictive Machine Learning Modeling

Olist 이커머스 AI 표준 파이프라인을 벤치마킹하여 4대 예측 및 추천 엔진을 모듈형으로 구축하였습니다.

| 엔진 번호 | AI 엔진 명칭 | 모델 알고리즘 | 입력 변수 (Features) | 출력 타깃 (Target) |
| :---: | :--- | :--- | :--- | :--- |
| **Engine 1** | **예상 총매출액 시뮬레이터** | Random Forest Regressor | 카테고리, 제안가격, 평점, 리뷰수 | 예상 총매출액($), 예상 누적 판매량 |
| **Engine 2** | **상품 조기 퇴출 위험 진단기** | Random Forest Classifier | 카테고리, 제안가격, 예상평점, 리뷰수 | 퇴출 위험 확률(%), 리스크 3단계 등급 |
| **Engine 3** | **고객 리뷰 실시간 감성 분석** | NLP Rule/Transformer Pipeline | 고객 영문 리뷰 원문 텍스트 | 긍정/부정 판정, 예측 신뢰도 점수 |
| **Engine 4** | **벤치마크 경쟁 상품 추천** | Rule-based Filtering & Ranking | 타깃 카테고리, 최대 예산 한도 | 최고 평점·바이럴 Top-5 벤치마크 리스트 |

---

## 6. Interactive Gradio Web Service Interface

현업 MD 및 이커머스 운영자가 신제품 기획 시 실시간으로 의사결정을 내릴 수 있도록 4단 탭 인터랙티브 대시보드를 구축하였습니다.

* **Tab 1 (Sales Engine)**: 카테고리와 출시 가격 슬라이더 조절 시 예상 매출액과 필요 판매 볼륨을 즉각 산출
* **Tab 2 (Churn Risk)**: 가격 및 타깃 평점 설정 시 조기 퇴출 위험도를 진단하여 과도한 가격 책정 사전 방지
* **Tab 3 (NLP Sentiment)**: 수집된 고객 VOC 및 리뷰를 입력하면 실시간 긍/부정 반응을 감별
* **Tab 4 (Recommendation)**: 예산 제약 조건 내에서 반드시 벤치마킹해야 할 상위 5개 경쟁사 핵심 상품 매칭

---

## 7. Data-Driven Business Strategies

1. **신규 입점 소싱 Sweet Spot ($25 ~ $50)**:
   * 럭셔리 라인 대비 소비 저항이 적으면서도 ANOVA 검정상 최고 수준의 만족도를 유지하는 최적의 마진 구간으로 확인됨
2. **사전 조기 경보(Early Warning) 프로세스 수립**:
   * Churn Model 기준 퇴출 위험도 50%를 초과하는 상품군은 입점 초기 재고 매입량을 40% 축소하고 얼리버드 리뷰 체험단 프로모션을 우선 배정
3. **리뷰 볼륨 중심의 랭킹 최적화**:
   * 단순 평점 중심 상품 노출을 지양하고, 바이럴 계수(Love)와 리뷰 수가 검증된 스테디셀러 위주의 추천 캐러셀 운영

---

## 8. Engineering Retrospective & Troubleshooting

* **Issue 1: Colab 런타임 재할당 시 메모리 휘발 (`NameError: name 'df_clean' is not defined`)**
  * **원인**: 가상 머신 연결 해제 후 하위 모델링 코드만 재실행할 때 전처리 객체 소실
  * **해결**: 데이터 파일 존재 여부를 자동 검사하고 무결성 정제 파이프라인을 단일 셀에서 원스톱으로 재구축하는 자동 복구 로직 구현
* **Issue 2: Colab 가상 머신 병렬 처리 데드락 (`n_jobs=-1` 무한 로딩)**
  * **원인**: 코랩 가상 CPU 환경에서 Scikit-learn의 멀티프로세싱 포크(fork) 충돌로 연산이 정지되는 현상
  * **해결**: `n_jobs=1` 명시적 단일 스레드 최적화 및 트리 수 경량화를 통해 모델 2종 학습 시간을 3초 이내로 단축
* **Issue 3: Gradio 터널링 프록시 대기 시간 지연 (`share=True` 프리징)**
  * **원인**: 외부 네트워크 환경에 따라 원격 터널링 포트(`frpc`) 바이너리 다운로드가 지연되는 병목 발생
  * **해결**: 개발 및 포트폴리오 캡처 단계에서는 인라인 렌더링(`demo.launch(inline=True)`)으로 즉시 서빙되도록 분기 처리
