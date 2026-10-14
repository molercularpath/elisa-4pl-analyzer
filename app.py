import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit


st.set_page_config(page_title="ELISA 4PL Analyzer", layout="wide")

st.title("ELISA 4PL Analyzer")
st.caption("4PL Standard Curve · Sample Concentration")


# 농도 증가에 따라 OD가 증가하거나 감소할 수 있는 4PL 모델
def four_pl(x, bottom, top, ec50, hill):
    x = np.asarray(x, dtype=float)
    return bottom + (top - bottom) / (1 + (x / ec50) ** hill)


def fit_4pl(concentrations, ods):
    """표준물질 데이터로 4PL 곡선을 맞추고 진단 정보를 반환합니다."""
    concentrations = np.asarray(concentrations, dtype=float)
    ods = np.asarray(ods, dtype=float)

    valid = (
        np.isfinite(concentrations)
        & np.isfinite(ods)
        & (concentrations > 0)
    )
    x = concentrations[valid]
    y = ods[valid]

    if len(x) < 4:
        raise ValueError("유효한 표준물질 데이터가 최소 4개 필요합니다.")
    if len(np.unique(x)) < 4:
        raise ValueError("서로 다른 농도값이 최소 4개 필요합니다.")

    bottom0 = float(np.min(y))
    top0 = float(np.max(y))
    ec500 = float(np.median(x))
    hill0 = 1.0 if y[-1] >= y[0] else -1.0

    params, _ = curve_fit(
        four_pl,
        x,
        y,
        p0=[bottom0, top0, ec500, hill0],
        bounds=(
            [-np.inf, -np.inf, 1e-12, -10],
            [np.inf, np.inf, np.inf, 10],
        ),
        maxfev=50000,
    )

    predicted = four_pl(x, *params)
    residuals = y - predicted

    rss = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1 - rss / ss_tot) if ss_tot > 0 else float("nan")

    # 농도순으로 정렬해 잔차 패턴을 보기 쉽게 함
    order = np.argsort(x)

    return (
        x[order],
        y[order],
        predicted[order],
        residuals[order],
        params,
        r_squared,
        rss,
    )

    if len(x) < 4:
        raise ValueError("유효한 표준물질 데이터가 최소 4개 필요합니다.")
    if len(np.unique(x)) < 4:
        raise ValueError("서로 다른 농도값이 최소 4개 필요합니다.")

    # 증가/감소 곡선 모두 대응하도록 초기값을 데이터에서 추정
    bottom0 = float(np.min(y))
    top0 = float(np.max(y))
    ec500 = float(np.median(x))
    hill0 = 1.0 if y[-1] >= y[0] else -1.0

    params, _ = curve_fit(
        four_pl,
        x,
        y,
        p0=[bottom0, top0, ec500, hill0],
        bounds=(
            [-np.inf, -np.inf, 1e-12, -10],
            [np.inf, np.inf, np.inf, 10],
        ),
        maxfev=50000,
    )

    predicted = four_pl(x, *params)
    ss_res = np.sum((y - predicted) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = (
        float(1 - ss_res / ss_tot)
        if ss_tot > 0
        else float("nan")
    )

    return x, y, params, r_squared


def inverse_4pl(od, params):
    """OD값을 4PL 모델의 농도로 역산합니다."""
    bottom, top, ec50, hill = params

    if np.isclose(hill, 0):
        return np.nan

    denominator = od - bottom
    numerator = top - bottom

    if np.isclose(denominator, 0) or np.isclose(numerator, 0):
        return np.nan

    ratio = numerator / denominator
    if ratio <= 0:
        return np.nan

    try:
        concentration = ec50 * (ratio - 1) ** (1 / hill)
    except (ValueError, OverflowError, ZeroDivisionError):
        return np.nan

    if not np.isfinite(concentration) or concentration < 0:
        return np.nan

    return float(concentration)


def read_excel(uploaded_file):
    """첫 번째 시트를 읽습니다."""
    return pd.read_excel(uploaded_file)


def to_excel_bytes(dataframe):
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False, sheet_name="Analysis")
    buffer.seek(0)
    return buffer.getvalue()


st.header("STEP 1. 표준곡선 Excel 업로드")
st.write("첫 번째 시트에 `Concentration`, `OD` 열을 넣어 주세요.")
standard_file = st.file_uploader(
    "표준곡선 파일",
    type=["xlsx", "xls"],
    key="standard",
)

st.header("STEP 2. 샘플 Excel 업로드")
st.write(
    "`Sample`, `OD1`, `OD2`, `OD3`, `Dilution_Factor` 열이 필요합니다. "
    "`OD1`~`OD3`은 반복 측정값이며, 희석하지 않은 샘플은 "
    "`Dilution_Factor`에 1을 입력하세요."
)
sample_file = st.file_uploader(
    "샘플 파일",
    type=["xlsx", "xls"],
    key="samples",
)

if standard_file and sample_file:
    try:
        standard_df = read_excel(standard_file)
        sample_df = read_excel(sample_file)

        standard_required = {"Concentration", "OD"}
        sample_required = {
            "Sample",
            "OD1",
            "OD2",
            "OD3",
            "Dilution_Factor",
        }

        missing_standard = standard_required - set(standard_df.columns)
        missing_sample = sample_required - set(sample_df.columns)

        if missing_standard:
            st.error(
                "표준곡선 파일에 필요한 열이 없습니다: "
                + ", ".join(sorted(missing_standard))
            )
            st.stop()

        if missing_sample:
            st.error(
                "샘플 파일에 필요한 열이 없습니다: "
                + ", ".join(sorted(missing_sample))
            )
            st.stop()

        concentrations = pd.to_numeric(
            standard_df["Concentration"], errors="coerce"
        ).to_numpy()
        standard_ods = pd.to_numeric(
            standard_df["OD"], errors="coerce"
        ).to_numpy()

(
    x,
    y,
    predicted,
    residuals,
    params,
    r_squared,
    rss,
) = fit_4pl(concentrations, standard_ods)


st.header("STEP 3. 분석 결과")

bottom, top, ec50, hill = params

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("R²", f"{r_squared:.5f}")
c2.metric("RSS", f"{rss:.8g}")
c3.metric("Bottom", f"{bottom:.5g}")
c4.metric("Top", f"{top:.5g}")
c5.metric("EC50", f"{ec50:.5g}")
c6.metric("Hill slope", f"{hill:.5g}")

# 표준점 범위와 파라미터 위치 진단
min_conc = float(np.min(x))
max_conc = float(np.max(x))
min_od = float(np.min(y))
max_od = float(np.max(y))

st.subheader("표준점 범위 및 곡선 형태")

range_col1, range_col2 = st.columns(2)
range_col1.write(
    f"**표준 농도 범위:** {min_conc:g} ~ {max_conc:g}"
)
range_col2.write(
    f"**관측 OD 범위:** {min_od:g} ~ {max_od:g}"
)

if min_conc <= ec50 <= max_conc:
    st.success("EC50이 표준 농도 범위 안에 있습니다.")
else:
    st.warning(
        "EC50이 표준 농도 범위 밖에 있습니다. "
        "상수 추정이 불안정할 수 있으니 표준점과 곡선을 확인하세요."
    )

if hill > 0:
    st.write("적합된 모델의 방향: **농도가 증가할수록 OD가 증가하는 형태**")
elif hill < 0:
    st.write("적합된 모델의 방향: **농도가 증가할수록 OD가 감소하는 형태**")
else:
    st.warning("Hill slope가 0에 가까워 곡선 형태를 확인해야 합니다.")

# 4PL 곡선과 표준점
fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(x, y, label="Standards", color="royalblue", zorder=3)

curve_x = np.geomspace(min_conc, max_conc, 300)
curve_y = four_pl(curve_x, *params)
ax.plot(curve_x, curve_y, color="darkorange", label="4PL fit")

ax.set_xscale("log")
ax.set_xlabel("Concentration")
ax.set_ylabel("OD")
ax.set_title("4PL Standard Curve")
ax.grid(True, alpha=0.25)
ax.legend()

st.pyplot(fig)
plt.close(fig)

        # 표준곡선 그래프
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(x, y, label="Standards", color="royalblue")

        curve_x = np.geomspace(np.min(x), np.max(x), 300)
        curve_y = four_pl(curve_x, *params)
        ax.plot(curve_x, curve_y, color="darkorange", label="4PL fit")

        ax.set_xscale("log")
        ax.set_xlabel("Concentration")
        ax.set_ylabel("OD")
        ax.set_title("4PL Standard Curve")
        ax.grid(True, alpha=0.25)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

st.subheader("표준점별 잔차")

residual_df = pd.DataFrame({
    "Concentration": x,
    "Observed_OD": y,
    "Predicted_OD": predicted,
    "Residual": residuals,
    "Absolute_Residual": np.abs(residuals),
    "Squared_Residual": residuals ** 2,
})

st.dataframe(residual_df, use_container_width=True)
st.caption(
    "잔차는 실측 OD − 예측 OD로 계산했어요. "
    "잔차 제곱의 합이 RSS입니다."
)

# 잔차를 농도에 대해 표시
resid_fig, resid_ax = plt.subplots(figsize=(8, 4))
resid_ax.axhline(0, color="black", linewidth=1)
resid_ax.scatter(x, residuals, color="purple")

resid_ax.set_xscale("log")
resid_ax.set_xlabel("Concentration")
resid_ax.set_ylabel("Residual (Observed - Predicted)")
resid_ax.set_title("Residual Pattern")
resid_ax.grid(True, alpha=0.25)

st.pyplot(resid_fig)
plt.close(resid_fig)

# 단순 패턴 경고: 잔차가 대부분 한쪽에 있는지 확인
positive_count = int(np.sum(residuals > 0))
negative_count = int(np.sum(residuals < 0))

if positive_count == 0 or negative_count == 0:
    st.warning(
        "잔차가 모두 같은 부호입니다. "
        "곡선이 데이터의 한쪽에 치우쳤는지 확인하세요."
    )
else:
    st.info(
        f"잔차 부호: 양수 {positive_count}개, 음수 {negative_count}개. "
        "잔차 그래프에 농도에 따른 뚜렷한 곡선형·연속적 패턴이 있는지 "
        "눈으로 확인하세요."
    )
    concentration_columns = [
    "OD1_Concentration",
    "OD2_Concentration",
    "OD3_Concentration",
]

# 각 샘플의 유효한 반복 농도 수
result["Valid_Replicates"] = result[concentration_columns].count(axis=1)

# 반복 OD가 표준물질의 관측 OD 범위 안에 있는지 점검
od_columns = ["OD1", "OD2", "OD3"]
sample_od_values = result[od_columns].apply(
    pd.to_numeric, errors="coerce"
)

result["OD_Within_Standard_Range"] = sample_od_values.apply(
    lambda row: row.dropna().between(min_od, max_od).all(),
    axis=1,
)

st.subheader("반복 측정값 진단")

st.dataframe(
    result[
        [
            "Sample",
            *od_columns,
            *concentration_columns,
            "Valid_Replicates",
            "Mean",
            "SD",
            "CV(%)",
            "OD_Within_Standard_Range",
        ]
    ],
    use_container_width=True,
)

st.caption(
    "`OD_Within_Standard_Range`는 샘플 OD가 표준물질의 관측 OD 최솟값~최댓값 "
    "안에 있는지 나타냅니다. 범위 안이라고 해서 자동으로 정량 신뢰성이 "
    "보장되는 것은 아니며, 표준곡선의 양 끝에서는 특히 주의해야 해요."
)


        # 각 OD 반복값을 농도로 환산하고 희석배수를 반영
        result = sample_df.copy()
        dilution = pd.to_numeric(
            result["Dilution_Factor"], errors="coerce"
        ).to_numpy()

        concentration_columns = []
        for od_column in ["OD1", "OD2", "OD3"]:
            od_values = pd.to_numeric(
                result[od_column], errors="coerce"
            ).to_numpy()

            calculated = []
            for od, factor in zip(od_values, dilution):
                if not np.isfinite(od) or not np.isfinite(factor) or factor <= 0:
                    calculated.append(np.nan)
                    continue

                measured_concentration = inverse_4pl(od, params)
                if np.isfinite(measured_concentration):
                    calculated.append(measured_concentration * factor)
                else:
                    calculated.append(np.nan)

            concentration_column = f"{od_column}_Concentration"
            result[concentration_column] = calculated
            concentration_columns.append(concentration_column)

        replicate_values = result[concentration_columns]
        result["Mean"] = replicate_values.mean(axis=1)
        result["SD"] = replicate_values.std(axis=1, ddof=1)
        result["CV(%)"] = np.where(
            result["Mean"] != 0,
            result["SD"] / result["Mean"] * 100,
            np.nan,
        )

        st.subheader("샘플 농도 및 통계")
        st.dataframe(result, use_container_width=True)

        st.download_button(
            label="분석 결과 Excel 다운로드",
            data=to_excel_bytes(result),
            file_name="elisa_4pl_analysis.xlsx",
            mime=(
                "application/vnd.openxmlformats-officedocument."
                "spreadsheetml.sheet"
            ),
        )

        st.caption(
            "농도 역산값이 비어 있다면 OD가 표준곡선의 유효 범위 밖인지 "
            "확인하세요. 표준물질 농도는 0보다 큰 값이어야 합니다."
        )

    except Exception as exc:
        st.error(f"분석 중 오류가 발생했습니다: {exc}")
else:
    st.info("분석하려면 표준곡선 파일과 샘플 파일을 모두 업로드하세요.")
