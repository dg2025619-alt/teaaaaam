import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# 기본 설정 (페이지 제목/아이콘)
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 뇌졸중 예측 실습실")

# -----------------------------
# 데이터 불러오기
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


df = load_data()

# -----------------------------
# 값의 종류(자료형) 한글로 표시하는 함수
# -----------------------------
def get_value_kind(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        return "정수형"
    elif pd.api.types.is_float_dtype(series):
        return "실수형"
    else:
        return "문자형(범주형)"


# -----------------------------
# 소개 화면
# -----------------------------
st.header("📋 데이터 소개")
st.write(
    "이 데이터는 여러 사람의 건강·생활 정보를 바탕으로 "
    "뇌졸중(stroke) 발생 여부를 살펴보기 위한 자료예요."
)

# 큰 숫자 카드 네 개
total_count = len(df)
col_count = df.shape[1]
stroke_count = int(df["stroke"].sum())
stroke_ratio = stroke_count / total_count * 100

card1, card2, card3, card4 = st.columns(4)
card1.metric("전체 사람 수", f"{total_count:,}명")
card2.metric("열 개수", f"{col_count}개")
card3.metric("뇌졸중(stroke=1) 인원", f"{stroke_count:,}명")
card4.metric("뇌졸중 비율", f"{stroke_ratio:.2f}%")

st.divider()

# -----------------------------
# 열 설명 표 (우리말 뜻은 비워 둠)
# -----------------------------
st.subheader("📑 열(컬럼) 설명")
st.caption("‘우리말 뜻’ 칸은 비어 있어요. 교재를 보고 직접 채워 넣어 보세요.")

info_rows = []
for col in df.columns:
    info_rows.append(
        {
            "열 이름": col,
            "우리말 뜻": "",
            "값의 종류": get_value_kind(df[col]),
            "빈 값 개수": int(df[col].isna().sum()),
        }
    )

info_df = pd.DataFrame(info_rows)

edited_info_df = st.data_editor(
    info_df,
    column_config={
        "우리말 뜻": st.column_config.TextColumn(
            "우리말 뜻", help="교재를 참고해서 이 열이 무엇을 뜻하는지 적어 보세요."
        )
    },
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    hide_index=True,
    use_container_width=True,
)

st.divider()

# -----------------------------
# 데이터 처음 다섯 줄
# -----------------------------
st.subheader("🔎 데이터 미리보기 (처음 5줄)")
st.dataframe(df.head(), use_container_width=True)

st.divider()

# -----------------------------
# 데이터 출처 적는 자리
# -----------------------------
st.subheader("📚 데이터 출처")
st.text_area(
    "교재에 나온 데이터 출처를 아래에 적어 보세요.",
    placeholder="여기에 데이터 출처를 입력하세요.",
    height=120,
)
