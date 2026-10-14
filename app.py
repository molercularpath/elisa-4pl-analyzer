import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.optimize import curve_fit


st.set_page_config(page_title="ELISA 4PL Analyzer", layout="wide")

st.title("ELISA 4PL Analyzer")
st.caption("4PL Standard Curve · Sample Concentration")


def four_pl(x, bottom, top, ec50, hill):
    """4PL 모델: 농도에 따라 증가형 또는 감소형 곡선을 표현합니다."""
    x = np.asarray(x, dtype=float)
    return bottom + (top - bottom) / (1 + (x / ec50) ** hill)


def fit_4pl(concentrations, ods):
    """표준물질 데이터에 4PL을 맞추고 잔차, RSS, R²를 반환합니다."""
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

    # 초기 추정값
    bottom0 = float(np.min(y))
    top0 = float(np.max(y))
    ec50_0 = float(np.median(x))

    # 농도 증가에 따른 OD 변화 방향으로 Hill 상수의 초기 부호 결정
    lowest_conc_od = y[np.argmin(x)]
    highest_conc_od = y[np.argmax(x)]
    hill0 = 1.0 if highest_conc_od >= lowest_conc_od else -1.0

    params, _ = curve_fit(
        four_pl,
        x,
        y,
        p0=[bottom0, top0, ec50_0, hill0],
        bounds=(
            [-np.inf, -np.inf, 1e-12, -10],
            [np.inf, np.inf, np.inf, 10],
        ),
        maxfev=50000,
    )

    predicted = four_pl(x, *params)
    residuals = y - predicted

    rss = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1 - rss / ss_tot) if ss_tot > 0 else float("nan")

    # 농도 오름차순으로 정렬
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


def inverse_4pl(od, params):
    """OD값을 4PL 모델의 농도로 역산합니다."""
    bottom, top, ec50, hill = params

    if not np.isfinite(od) or np.isclose(hill, 0):
        return np.nan

    numerator = top - bottom
    denominator = od - bottom

    if np.isclose(numerator, 0) or np.isclose(denominator, 0):
        return np.nan

    ratio = numerator / denominator

    if not np.isfinite(ratio) or ratio <= 0:
        return np.nan

    base = ratio - 1

    # 실수 범위의 농도값이 나오지 않는 경우
    if base < 0:
        return np.nan

    try:
        concentration = ec50 * base ** (1 / hill)
    except (ValueError, OverflowError, ZeroDivisionError):
        return np.nan

    if not np.isfinite(concentration) or concentration < 0:
        return np.nan

    return float(concentration)


def read_excel(uploaded_file):
    """업로드한 Excel 파일의 첫 번째 시트를 읽습니다."""
    return pd.read_excel(uploaded_file)


def to_excel_bytes(dataframe):
    """결과 DataFrame을 Excel 파일 바이트로 변환합니다."""
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
    "`OD1`~`OD3`은 반복 측정값입니다. 희석하지 않은 샘플은 "
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

        # 표준 데이터 숫자 변환
        concentrations = pd.to_numeric(
            standard_df["Concentration"],
            errors="coerce",
        ).to_numpy()

        standard_ods = pd.to_numeric(
            standard_df["OD"],
            errors="coerce",
        ).to_numpy()

        # 4PL 적합
        (
            x,
            y,
            predicted,
            residuals,
            params,
            r_squared,
            rss,
        ) = fit_4pl(concentrations, standard_ods)

        bottom, top, ec50, hill = params

        st.header("STEP 3. 분석 결과")

        # 적합 지표
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("R²", f"{r_squared:.5f}")
        c2.metric("RSS", f"{rss:.8g}")
        c3.metric("Bottom", f"{bottom:.5g}")
        c4.metric("Top", f"{top:.5g}")
        c5.metric("EC50", f"{ec50:.5g}")
        c6.metric("Hill slope", f"{hill:.5g}")

        # 표준 범위와 모델 형태
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
                "상수 추정과 곡선 형태를 확인하세요."
            )

        if hill > 0:
            st.write(
                "적합된 모델 방향: **농도가 증가할수록 OD가 증가하는 형태**"
            )
        elif hill < 0:
            st.write(
                "적합된 모델 방향: **농도가 증가할수록 OD가 감소하는 형태**"
            )
        else:
            st.warning("Hill slope가 0에 가까워 곡선 형태를 확인해야 합니다.")

        # 표준곡선 그래프
        fig, ax = plt.subplots(figsize=(8, 5))

        ax.scatter(
            x,
            y,
            label="Standards",
            color="royalblue",
            zorder=3,
        )

        curve_x = np.geomspace(min_conc, max_conc, 300)
        curve_y = four_pl(curve_x, *params)

        ax.plot(
            curve_x,
            curve_y,
            color="darkorange",
            label="4PL fit",
        )

        ax.set_xscale("log")
        ax.set_xlabel("Concentration")
        ax.set_ylabel("OD")
        ax.set_title("4PL Standard Curve")
        ax.grid(True, alpha=0.25)
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

        # 표준점별 잔차표
        st.subheader("표준점별 잔차")

        residual_df = pd.DataFrame({
            "Concentration": x,
            "Observed_OD": y,
            "Predicted_OD": predicted,
            "Residual": residuals,
            "Absolute_Residual": np.abs(residuals),
            "Squared_Residual": residuals**2,
        })

        st.dataframe(residual_df, use_container_width=True)
        st.caption(
            "잔차는 실측 OD − 예측 OD이며, 잔차 제곱의 합이 RSS입니다."
        )

        # 잔차 패턴 그래프
        residual_fig, residual_ax = plt.subplots(figsize=(8, 4))

        residual_ax.axhline(0, color="black", linewidth=1)
        residual_ax.scatter(x, residuals, color="purple")

        residual_ax.set_xscale("log")
        residual_ax.set_xlabel("Concentration")
        residual_ax.set_ylabel("Residual (Observed - Predicted)")
        residual_ax.set_title("Residual Pattern")
        residual_ax.grid(True, alpha=0.25)

        st.pyplot(residual_fig)
        plt.close(residual_fig)

        positive_count = int(np.sum(residuals > 0))
        negative_count = int(np.sum(residuals < 0))

        if positive_count == 0 or negative_count == 0:
            st.warning(
                "잔차가 모두 같은 부호입니다. "
                "곡선이 데이터의 한쪽으로 치우쳤는지 확인하세요."
            )
        else:
            st.info(
                f"잔차 부호: 양수 {positive_count}개, "
                f"음수 {negative_count}개. "
                "잔차 그래프에 농도에 따른 뚜렷한 패턴이 있는지 확인하세요."
            )

        # 샘플 농도 계산
        result = sample_df.copy()

        dilution = pd.to_numeric(
            result["Dilution_Factor"],
            errors="coerce",
        ).to_numpy()

        od_columns = ["OD1", "OD2", "OD3"]
        concentration_columns = []

        for od_column in od_columns:
            od_values = pd.to_numeric(
                result[od_column],
                errors="coerce",
            ).to_numpy()

            calculated = []

            for od, factor in zip(od_values, dilution):
                if (
                    not np.isfinite(od)
                    or not np.isfinite(factor)
                    or factor <= 0
                ):
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

        # 농도 반복값의 통계
        replicate_values = result[concentration_columns]

        result["Mean"] = replicate_values.mean(axis=1)
        result["SD"] = replicate_values.std(axis=1, ddof=1)
        result["CV(%)"] = np.where(
            result["Mean"] != 0,
            result["SD"] / result["Mean"] * 100,
            np.nan,
        )

        result["Valid_Replicates"] = replicate_values.count(axis=1)

        # OD가 표준물질에서 관측된 OD 범위 안인지 확인
        sample_od_values = result[od_columns].apply(
            pd.to_numeric,
            errors="coerce",
        )

        result["OD_Within_Standard_Range"] = sample_od_values.apply(
            lambda row: (
                row.dropna().between(min_od, max_od).all()
                if row.notna().any()
                else False
            ),
            axis=1,
        )

        st.subheader("샘플 농도 및 반복 측정값 진단")
        st.dataframe(result, use_container_width=True)

        st.caption(
            "`OD_Within_Standard_Range`는 샘플 OD가 표준물질에서 관측된 "
            "OD 범위 안에 있는지 나타냅니다. 범위 안에 있다는 사실만으로 "
            "정량의 신뢰성이 보장되지는 않으며, 분석법의 정량 범위도 "
            "별도로 확인해야 합니다."
        )

        # Excel 다운로드
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
            "결과가 비어 있다면 OD가 모델에서 역산 가능한 범위인지, "
            "희석배수가 0보다 큰지 확인하세요."
        )

    except Exception as exc:
        st.error(f"분석 중 오류가 발생했습니다: {exc}")

else:
    st.info("분석하려면 표준곡선 파일과 샘플 파일을 모두 업로드하세요.")
