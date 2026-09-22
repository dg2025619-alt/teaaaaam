import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import resample

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(
    page_title="뇌졸중 예측 실습실 - 분류 모델",
    page_icon="🧩",
    layout="wide",
)

st.title("🧩 분류 모델 만들기")

# -----------------------------
# 데이터 불러오기 (첫 화면과 동일)
# -----------------------------
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/stroke.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    return df


df = load_data()

FEATURE_NAME_MAP = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
FEATURE_OPTIONS = list(FEATURE_NAME_MAP.keys())
DEFAULT_FEATURES = [c for c in FEATURE_OPTIONS if c != "bmi"]

# -----------------------------
# 1. 입력 속성 고르기
# -----------------------------
st.header("1️⃣ 입력 속성 고르기")

selected_features = st.multiselect(
    "모델이 사용할 속성을 골라 주세요.",
    options=FEATURE_OPTIONS,
    default=DEFAULT_FEATURES,
    format_func=lambda x: FEATURE_NAME_MAP[x],
)

if len(selected_features) < 2:
    st.warning("속성을 두 개 이상 골라야 모델을 만들 수 있어요.")
    st.stop()

# -----------------------------
# 데이터 준비: 번호순 정렬 → 10명씩 묶어 앞 3명은 테스트용
# -----------------------------
df_sorted = df.sort_values("id").reset_index(drop=True)
positions = [i % 10 for i in range(len(df_sorted))]
df_sorted = df_sorted.copy()
df_sorted["_pos"] = positions

test_df = df_sorted[df_sorted["_pos"] < 3].drop(columns="_pos").reset_index(drop=True)
train_df = df_sorted[df_sorted["_pos"] >= 3].drop(columns="_pos").reset_index(drop=True)

# bmi 결측치는 훈련용의 중앙값으로 채움 (bmi를 고른 경우에만)
if "bmi" in selected_features:
    bmi_median = train_df["bmi"].median()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["bmi"] = train_df["bmi"].fillna(bmi_median)
    test_df["bmi"] = test_df["bmi"].fillna(bmi_median)

# -----------------------------
# 훈련용 크기 맞추기 (적은 쪽을 늘려서 맞춤, 테스트용은 그대로 둠)
# -----------------------------
train_majority_label = train_df["stroke"].value_counts().idxmax()
df_majority = train_df[train_df["stroke"] == train_majority_label]
df_minority = train_df[train_df["stroke"] != train_majority_label]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42,
)
train_balanced = pd.concat([df_majority, df_minority_upsampled]).reset_index(drop=True)

X_train = train_balanced[selected_features]
y_train = train_balanced["stroke"]
X_test = test_df[selected_features]
y_test = test_df["stroke"]

# -----------------------------
# 모델 학습
# -----------------------------
log_model = LogisticRegression(random_state=42, max_iter=1000)
log_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(
    max_depth=3, min_samples_leaf=5, random_state=42
)
tree_model.fit(X_train, y_train)


def get_accuracies(model):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    return accuracy_score(y_train, train_pred), accuracy_score(y_test, test_pred)


log_train_acc, log_test_acc = get_accuracies(log_model)
tree_train_acc, tree_test_acc = get_accuracies(tree_model)

# 다수결(입력 무시) 기준 모델: 훈련용(원래, 크기 맞추기 전)에서 많은 쪽으로만 답함
baseline_label = train_df["stroke"].value_counts().idxmax()
baseline_train_acc = accuracy_score(train_df["stroke"], [baseline_label] * len(train_df))
baseline_test_acc = accuracy_score(test_df["stroke"], [baseline_label] * len(test_df))

# -----------------------------
# 2. 모델 정확도 비교
# -----------------------------
st.header("2️⃣ 모델 정확도 비교")


def show_acc_card(col, title, train_acc, test_acc):
    with col:
        st.metric(title, f"{test_acc * 100:.1f}%")
        st.caption(f"훈련 정확도 {train_acc * 100:.1f}% · 테스트 정확도 {test_acc * 100:.1f}%")


acc_col1, acc_col2, acc_col3 = st.columns(3)
show_acc_card(acc_col1, "로지스틱 회귀(확률로 답하는 모델)", log_train_acc, log_test_acc)
show_acc_card(acc_col2, "의사결정트리(질문으로 답하는 모델)", tree_train_acc, tree_test_acc)
show_acc_card(
    acc_col3,
    "입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델",
    baseline_train_acc,
    baseline_test_acc,
)

st.divider()

# -----------------------------
# 3. 산점도와 경계선으로 살펴보기
# -----------------------------
st.header("3️⃣ 산점도와 경계선으로 살펴보기")

axis_col1, axis_col2 = st.columns(2)
with axis_col1:
    x_feature = st.selectbox(
        "가로축으로 쓸 속성",
        selected_features,
        index=0,
        format_func=lambda x: FEATURE_NAME_MAP[x],
    )
with axis_col2:
    remaining_features = [f for f in selected_features if f != x_feature]
    y_feature = st.selectbox(
        "세로축으로 쓸 속성",
        remaining_features,
        index=0,
        format_func=lambda x: FEATURE_NAME_MAP[x],
    )

other_features = [f for f in selected_features if f not in (x_feature, y_feature)]
fixed_values = {f: float(test_df[f].median()) for f in other_features}

if fixed_values:
    fixed_desc = ", ".join(
        f"{FEATURE_NAME_MAP[f]}={v:.2f}" for f, v in fixed_values.items()
    )
    st.write(f"그림에 쓰지 않은 속성은 테스트 데이터의 중앙값으로 고정했어요: {fixed_desc}")
else:
    st.write("고른 속성이 두 개뿐이라 따로 고정한 속성은 없어요.")

# 좌표 범위 계산 (여유 공간 포함)
x_min, x_max = float(test_df[x_feature].min()), float(test_df[x_feature].max())
y_min, y_max = float(test_df[y_feature].min()), float(test_df[y_feature].max())
pad_x = (x_max - x_min) * 0.05 if x_max > x_min else 1.0
pad_y = (y_max - y_min) * 0.05 if y_max > y_min else 1.0
x_range = [x_min - pad_x, x_max + pad_x]
y_range = [y_min - pad_y, y_max + pad_y]

# ---- 의사결정트리가 나눈 칸(배경) 계산 ----
grid_n = 60
xs_grid = [x_range[0] + i * (x_range[1] - x_range[0]) / (grid_n - 1) for i in range(grid_n)]
ys_grid = [y_range[0] + i * (y_range[1] - y_range[0]) / (grid_n - 1) for i in range(grid_n)]

grid_records = []
for gy in ys_grid:
    for gx in xs_grid:
        rec = {}
        for f in selected_features:
            if f == x_feature:
                rec[f] = gx
            elif f == y_feature:
                rec[f] = gy
            else:
                rec[f] = fixed_values[f]
        grid_records.append(rec)

grid_df = pd.DataFrame(grid_records)[selected_features]
grid_pred = tree_model.predict(grid_df)

z_grid = []
idx = 0
for _ in ys_grid:
    row = [int(v) for v in grid_pred[idx: idx + grid_n]]
    z_grid.append(row)
    idx += grid_n

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        x=xs_grid,
        y=ys_grid,
        z=z_grid,
        zmin=0,
        zmax=1,
        colorscale=[[0, "#dbe9ff"], [1, "#ffd9d9"]],
        showscale=False,
        opacity=0.35,
        hoverinfo="skip",
    )
)

# ---- 테스트 데이터 점 찍기 ----
test_plot_df = test_df.copy()
test_plot_df["실제 뇌졸중 여부"] = test_plot_df["stroke"].map({0: "아님", 1: "뇌졸중"})

scatter_fig = px.scatter(
    test_plot_df,
    x=x_feature,
    y=y_feature,
    color="실제 뇌졸중 여부",
    color_discrete_map={"아님": "#1f77b4", "뇌졸중": "#d62728"},
)
for trace in scatter_fig.data:
    fig.add_trace(trace)

# ---- 로지스틱 회귀 경계선(확률 0.5) 계산 ----
feature_index = {f: i for i, f in enumerate(selected_features)}
coef = log_model.coef_[0]
intercept = log_model.intercept_[0]

cx = coef[feature_index[x_feature]]
cy = coef[feature_index[y_feature]]
constant = intercept
for f in other_features:
    constant += coef[feature_index[f]] * fixed_values[f]

line_note = None

if abs(cy) > 1e-9:
    xs_line = [x_range[0] + i * (x_range[1] - x_range[0]) / 199 for i in range(200)]
    ys_line = [-(cx * x + constant) / cy for x in xs_line]
    inside = [y for y in ys_line if y_range[0] <= y <= y_range[1]]
    if inside:
        fig.add_trace(
            go.Scatter(
                x=xs_line,
                y=ys_line,
                mode="lines",
                name="로지스틱 회귀 경계선(확률 0.5)",
                line=dict(color="black", dash="dash"),
            )
        )
    else:
        line_note = "로지스틱 회귀의 경계선이 이 그림의 세로축 범위 밖에 있어요."
elif abs(cx) > 1e-9:
    x_line = -constant / cx
    if x_range[0] <= x_line <= x_range[1]:
        fig.add_trace(
            go.Scatter(
                x=[x_line, x_line],
                y=y_range,
                mode="lines",
                name="로지스틱 회귀 경계선(확률 0.5)",
                line=dict(color="black", dash="dash"),
            )
        )
    else:
        line_note = "로지스틱 회귀의 경계선이 이 그림의 가로축 범위 밖에 있어요."
else:
    line_note = "선택한 두 속성만으로는 경계선의 기울기를 구할 수 없어요."

fig.update_layout(
    title=f"{FEATURE_NAME_MAP[x_feature]} vs {FEATURE_NAME_MAP[y_feature]}",
    xaxis_title=FEATURE_NAME_MAP[x_feature],
    yaxis_title=FEATURE_NAME_MAP[y_feature],
    xaxis=dict(range=x_range),
    yaxis=dict(range=y_range),
    legend_title_text="실제 뇌졸중 여부",
)

st.plotly_chart(fig, use_container_width=True)

if line_note:
    st.info(line_note)

st.caption("옅은 파란색/빨간색 배경은 의사결정트리가 나눈 칸(예측 영역)을 나타내요.")

st.divider()

# -----------------------------
# 4. 의사결정트리, 가지 그림으로 보기
# -----------------------------
st.header("4️⃣ 의사결정트리, 가지 그림으로 보기")

tree_ = tree_model.tree_
feature_order = selected_features


def build_dot():
    lines = [
        "digraph Tree {",
        'node [shape=box, style="rounded,filled", fontsize=11];',
        'edge [fontsize=11];',
    ]
    leaf_count = 0
    leaf_no_count = 0
    used_features = set()

    def recurse(node_id):
        nonlocal leaf_count, leaf_no_count

        n_samples = int(tree_.n_node_samples[node_id])
        values = tree_.value[node_id][0]
        stroke_count = int(values[1]) if len(values) > 1 else 0
        ratio = (stroke_count / n_samples * 100) if n_samples > 0 else 0.0

        is_leaf = tree_.feature[node_id] == -2

        if is_leaf:
            leaf_count += 1
            predicted = "뇌졸중" if values[1] > values[0] else "아님"
            if predicted == "아님":
                leaf_no_count += 1
                fill = "#cfe8ff"
            else:
                fill = "#ffd0d0"
            label = (
                f"인원 {n_samples}명\\n뇌졸중 {stroke_count}명 ({ratio:.1f}%)\\n답: {predicted}"
            )
            lines.append(f'n{node_id} [label="{label}", fillcolor="{fill}"];')
        else:
            feat_idx = tree_.feature[node_id]
            feat_name = feature_order[feat_idx]
            used_features.add(feat_name)
            threshold = tree_.threshold[node_id]
            question = f"{FEATURE_NAME_MAP[feat_name]} <= {threshold:.2f} 인가요?"
            label = f"{question}\\n인원 {n_samples}명\\n뇌졸중 {stroke_count}명 ({ratio:.1f}%)"
            lines.append(f'n{node_id} [label="{label}", fillcolor="#f5f5f5"];')

            left = tree_.children_left[node_id]
            right = tree_.children_right[node_id]
            recurse(left)
            recurse(right)
            lines.append(f'n{node_id} -> n{left} [label="예"];')
            lines.append(f'n{node_id} -> n{right} [label="아니요"];')

    recurse(0)
    lines.append("}")
    return "\n".join(lines), leaf_count, leaf_no_count, used_features


dot_str, leaf_count, leaf_no_count, used_features = build_dot()
st.graphviz_chart(dot_str)

st.markdown(
    f"- 답을 내는 마디(잎)는 모두 **{leaf_count}칸**이고, 그중 **{leaf_no_count}칸**이 '아님'이라고 답해요."
)
for f in selected_features:
    used_text = "실제로 물었어요" if f in used_features else "묻지 않았어요"
    st.markdown(f"- {FEATURE_NAME_MAP[f]}: 이 나무는 {used_text}.")
