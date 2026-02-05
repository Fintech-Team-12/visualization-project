# 🏠 Apartment Sales Visualization & Analysis

**대한민국 아파트 실거래·소득 데이터를 기반으로 한 시각화 및 내 집 마련 시뮬레이션 프로젝트**

이 프로젝트는 대한민국 아파트 실거래 데이터와 개인 소득 데이터를 결합하여  
**지역별 아파트 거래 흐름**, **소득 대비 주거 접근성**,  
그리고 **내 집 마련까지 걸리는 시간**을 직관적으로 보여주는  
Streamlit 기반 데이터 분석 대시보드입니다.

지도 시각화, 지수 비교, 시뮬레이션을 통해  
> *“집값과 소득의 격차가 실제 삶에 어떤 의미인지”*  
를 체감하는 것을 목표로 합니다.

---

## ✨ 주요 기능

### 📍 지도 기반 시각화
- 법정동별 아파트 거래량
- 광역시도별 거래금액·평당가격 중앙값

### 📊 소득 vs 아파트 가격 지수 비교
- 2010년 기준 지수화
- 연도별 격차 시각화 (Dumbbell Chart)

### 📏 구매 가능 면적 & 1평 구매에 필요한 기간

### 🧮 내 집 마련 시뮬레이터
- 저축률, 연봉 상승률 조정
- 외벌이 / 맞벌이 시나리오 비교

### 🖥️ Streamlit 기반 인터랙티브 대시보드

---

## 🚀 실행 방법

이 프로젝트는 **pip 기반 실행**과 **Docker 기반 실행**  
두 가지 방식 모두를 지원합니다.

---

## 1️⃣ 로컬 실행 (pip 기반)

Python 환경에서 바로 실행하는 방식입니다.

### 1. 의존성 설치

```bash
pip install -r requirements.txt
````

### 2. 앱 실행

```bash
streamlit run app.py
```

### 3. 브라우저 접속

```
http://localhost:8501
```

> 💡 **Python 3.11** 환경을 권장합니다.

---

## 2️⃣ Docker 기반 실행 (배포 / 재현용 추천)

환경 차이 없이 동일하게 실행하고 싶을 때 가장 안정적인 방법입니다.

### 1. Docker 이미지 빌드

```bash
docker build --no-cache -t visual_project .
```

### 2. 컨테이너 실행

```bash
docker run --rm -p 8501:8501 visual_project
```

### 3. 브라우저 접속

```
http://localhost:8501
```

* `--rm` 옵션은 컨테이너 종료 시 자동 삭제를 의미합니다.
* 로컬 테스트 및 배포 환경에서 컨테이너 관리를 깔끔하게 해줍니다.

---

## 🌐 Streamlit Cloud 배포 버전

별도의 설치 없이 바로 체험할 수 있는 **공식 배포 버전**입니다.

👉 **Live Demo**
🔗 [https://apartment-sales-visualization-analysis.streamlit.app/](https://apartment-sales-visualization-analysis.streamlit.app/)

* Streamlit Cloud 환경에서 실행
* 로컬 / Docker 버전과 동일한 기능 제공
* 데이터 시각화 및 시뮬레이션 전체 포함

---

## 📁 프로젝트 구조

```text
.
├── app.py                 # Streamlit 메인 애플리케이션
├── requirements.txt       # Python 의존성
├── Dockerfile             # Docker 배포 설정
├── .dockerignore
├── data/                  # 데이터 파일
│   ├── 1인당_개인소득.csv
│   ├── 시도_apart_2010_data.csv
│   ├── 시도_apart_2015_data.csv
│   ├── 시도_apart_2020_data.csv
│   ├── 시도_apart_2025_data.csv
│   ├── 시도_apart_all_data.csv
│   ├── 대한민국_광역자치단체_경계.geojson
│   ├── ...
│   └── 법정동주소 거래량 데이터.csv
└── README.md
```

---

## 🛠️ 기술 스택

* **Python**
* **Streamlit**
* **Pandas / NumPy**
* **Plotly**
* **PyDeck**
* **Docker**

---

## 📮 마무리

이 프로젝트는
**“집값 문제를 데이터로 이해하고, 개인의 현실에 대입해보는 시도”**입니다.

피드백, 개선 아이디어, 확장 제안은 언제든 환영합니다 🙂
- 서울대학교 빅데이터 AI 핀테크 고급 전문가 과정 12기 시각화웹개발 프로젝트 1조