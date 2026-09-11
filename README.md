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
1. [Project Overview & System Architecture](#sec-1)
2. [Data Engineering & Preprocessing Pipeline](#sec-2)
3. [Deep-Dive Exploratory Data Analysis (EDA)](#sec-3)
4. [Statistical Hypothesis Testing (One-way ANOVA)](#sec-4)
5. [Predictive Machine Learning Modeling](#sec-5)
6. [Interactive Gradio Web Service Interface](#sec-6)
7. [Data-Driven Business Strategies](#sec-7)
8. [Engineering Retrospective & Troubleshooting](#sec-8)

---

<a id="sec-1"></a>
## 1. Project Overview & System Architecture

### 1.1 비즈니스 문제 정의 (Business Problem)
이커머스 뷰티 플랫폼에서 **"고가 프리미엄 제품일수록 고객 만족도(평점)가 유의미하게 더 높은가?"**는 MD 소싱과 프로모션 예산 배분의 핵심 의사결정 기준입니다.
* **가설 1**: 제품 가격이 높을수록 브랜드 충성도와 원료 품질에 의해 고객 평점(Rating)이 선형적으로 비례하여 상승할 것이다.
* **가설 2**: 플랫폼 내 등록 상품 수(SKU)가 많은 메이저 카테고리가 실제 고객의 바이럴 반응(Love/위시리스트 수)에서도 동일하게 높은 점유율을 차지할 것이다.

### 1.2 엔드투엔드 아키텍처 다이어그램 (Pipeline)

```mermaid
flowchart TD
    A[Sephora Raw Dataset: 9,168건] --> B[Data Cleansing: 0점 노이즈 398건 제거]
    B --> C[Feature Engineering: 4-Tier 가격 구간화]
    C --> D[EDA: 카테고리별 상관관계 및 바이럴 분석]
    C --> E[Statistical Testing: One-way ANOVA F=42.95, p<0.001]
    C --> F[Machine Learning: 하트수 및 평점 예측 Random Forest/XGBoost]
    E & F --> G[Interactive Web App: Gradio 기반 실시간 MD 시뮬레이터]
    G --> H[Actionable Business Strategy: 소싱 Sweet Spot 도출]
