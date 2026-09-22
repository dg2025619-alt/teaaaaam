import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 탐색",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 데이터 탐색")

# -----------------------------
# 데이터 불러오기 (첫 화면과 동일)
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


df = load_data()

# -----------------------------
# 1. 나이 / 평균 혈당 분포 히스토그램
# -----------------------------
st.header("1️⃣ 나이와 평균 혈당의 분포")

hist_col1, hist_col2 = st.columns(2)

with hist_col1:
    fig_age_hist = px.histogram(
        df, x="age", nbins=30, title="나이 분포"
    )
    fig_age_hist.update_layout(xaxis_title="나이", yaxis_title="사람 수")
    st.plotly_chart(fig_age_hist, use_container_width=True)

with hist_col2:
    fig_glucose_hist = px.histogram(
        df, x="avg_glucose_level", nbins=30, title="평균 혈당 분포"
    )
    fig_glucose_hist.update_layout(xaxis_title="평균 혈당", yaxis_title="사람 수")
    st.plotly_chart(fig_glucose_hist, use_container_width=True)

st.divider()

# -----------------------------
# 2. 뇌졸중 여부에 따른 나이 / 평균 혈당 상자그림 + 평균값 표
# -----------------------------
st.header("2️⃣ 뇌졸중 여부에 따른 나이와 평균 혈당 비교")

df_box = df.copy()
df_box["뇌졸중 여부"] = df_box["stroke"].map({0: "겪지 않음", 1: "겪음"})

box_col1, box_col2 = st.columns(2)

with box_col1:
    fig_age_box = px.box(
        df_box, x="뇌졸중 여부", y="age", title="뇌졸중 여부별 나이"
    )
    fig_age_box.update_layout(xaxis_title="", yaxis_title="나이")
    st.plotly_chart(fig_age_box, use_container_width=True)

with box_col2:
    fig_glucose_box = px.box(
        df_box, x="뇌졸중 여부", y="avg_glucose_level", title="뇌졸중 여부별 평균 혈당"
    )
    fig_glucose_box.update_layout(xaxis_title="", yaxis_title="평균 혈당")
    st.plotly_chart(fig_glucose_box, use_container_width=True)

mean_table = (
    df_box.groupby("뇌졸중 여부")[["age", "avg_glucose_level"]]
    .mean()
    .rename(columns={"age": "평균 나이", "avg_glucose_level": "평균 혈당"})
    .reset_index()
)
st.subheader("그룹별 평균값")
st.dataframe(mean_table, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# 3. 고혈압 / 심장병 유무에 따른 뇌졸중 비율 막대그래프
# -----------------------------
st.header("3️⃣ 고혈압 · 심장병 유무에 따른 뇌졸중 비율")

bar_col1, bar_col2 = st.columns(2)

with bar_col1:
    hyper_ratio = (
        df.groupby("hypertension")["stroke"].mean().reset_index()
    )
    hyper_ratio["hypertension"] = hyper_ratio["hypertension"].map(
        {0: "고혈압 없음", 1: "고혈압 있음"}
    )
    hyper_ratio["stroke"] = hyper_ratio["stroke"] * 100
    fig_hyper = px.bar(
        hyper_ratio,
        x="hypertension",
        y="stroke",
        title="고혈압 유무에 따른 뇌졸중 비율",
        text_auto=".2f",
    )
    fig_hyper.update_layout(xaxis_title="", yaxis_title="뇌졸중 비율(%)")
    st.plotly_chart(fig_hyper, use_container_width=True)

with bar_col2:
    heart_ratio = (
        df.groupby("heart_disease")["stroke"].mean().reset_index()
    )
    heart_ratio["heart_disease"] = heart_ratio["heart_disease"].map(
        {0: "심장병 없음", 1: "심장병 있음"}
    )
    heart_ratio["stroke"] = heart_ratio["stroke"] * 100
    fig_heart = px.bar(
        heart_ratio,
        x="heart_disease",
        y="stroke",
        title="심장병 유무에 따른 뇌졸중 비율",
        text_auto=".2f",
    )
    fig_heart.update_layout(xaxis_title="", yaxis_title="뇌졸중 비율(%)")
    st.plotly_chart(fig_heart, use_container_width=True)

st.divider()

# -----------------------------
# 4. bmi 결측 여부에 따른 뇌졸중 비율 비교
# -----------------------------
st.header("4️⃣ 체질량지수(bmi) 결측 여부와 뇌졸중 비율")

bmi_missing_count = int(df["bmi"].isna().sum())
bmi_missing_stroke_ratio = df.loc[df["bmi"].isna(), "stroke"].mean() * 100
overall_stroke_ratio = df["stroke"].mean() * 100

bmi_compare_table = pd.DataFrame(
    {
        "구분": ["전체", "bmi 결측 인원"],
        "인원 수": [len(df), bmi_missing_count],
        "뇌졸중 비율(%)": [
            round(overall_stroke_ratio, 2),
            round(bmi_missing_stroke_ratio, 2),
        ],
    }
)
st.dataframe(bmi_compare_table, use_container_width=True, hide_index=True)

st.divider()

# -----------------------------
# 5. 흡연 상태별 사람 수
# -----------------------------
st.header("5️⃣ 흡연 상태별 사람 수")

smoking_table = (
    df["smoking_status"]
    .value_counts()
    .reset_index()
)
smoking_table.columns = ["흡연 상태", "사람 수"]
st.dataframe(smoking_table, use_container_width=True, hide_index=True)
