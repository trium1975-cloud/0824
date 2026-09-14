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

