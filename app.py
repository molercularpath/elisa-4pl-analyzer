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
    """표준물질 데이터로 4PL 곡선을 맞춥니다."""
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

        x, y, params, r_squared = fit_4pl(concentrations, standard_ods)

        st.header("STEP 3. 분석 결과")

        bottom, top, ec50, hill = params
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("R²", f"{r_squared:.5f}")
        c2.metric("Bottom", f"{bottom:.5g}")
        c3.metric("Top", f"{top:.5g}")
        c4.metric("EC50", f"{ec50:.5g}")
        c5.metric("Hill slope", f"{hill:.5g}")

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