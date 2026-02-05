import streamlit as st
import pandas as pd
import pydeck as pdk
import json
import plotly.graph_objects as go
from pathlib import Path
import re
import unicodedata
from PIL import Image   
import os 
import plotly.express as px
import numpy as np 



# =========================
# 0) Streamlit 설정
# =========================
st.set_page_config(page_title="내 집 마련의 꿈", layout="wide")

# =========================
# 1) 경로/파일 탐색 (mac 한글 NFC/NFD 문제 회피)
# =========================
BASE_DIR = Path(__file__).resolve().parent
# BASE_DIR = BASE_DIR + "/data"

def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


files_in_dir = sorted([p.name for p in BASE_DIR.iterdir()])

# ✅ 아파트 파일: 시도_apart_YYYY_data.csv
apt_pat = re.compile(r"^시도_apart_(\d{4})_data\.csv$")
apart_files = sorted([str(BASE_DIR / fn) for fn in files_in_dir if apt_pat.match(nfc(fn))])

# ✅ 소득 파일
WAGE_PATH = str(BASE_DIR / "1인당_개인소득.csv")

if not apart_files:
    st.error(
        "❌ 아파트 데이터 파일을 찾지 못했습니다.\n\n"
        f"찾는 위치: {BASE_DIR}\n\n"
        "필요 파일 예시:\n"
        "  시도_apart_2010_data.csv, 시도_apart_2015_data.csv, 시도_apart_2020_data.csv, 시도_apart_2025_data.csv"
    )
    st.stop()

if not Path(WAGE_PATH).exists():
    st.error(
        "❌ 소득 데이터 파일(1인당_개인소득.csv)을 찾지 못했습니다.\n\n"
        f"찾는 위치: {BASE_DIR}"
    )
    st.stop()

#윤재파트=====
# =========================================================
# 📍 (추가) 지도 시각화 섹션: 법정동 거래량 + 광역시도 트렌드
# - ✅ import/set_page_config/BASE_DIR 중복 없음
# - ✅ 모든 파일 경로 BASE_DIR 기준
# - ✅ 'all' 파일 없이 2010/2015/2020/2025로 'all' 자동 생성(평균)
# - ✅ 컬럼명 오타/불일치 방어
# =========================================================

# 필요한 모듈은 기준 코드 상단 import에 추가되어 있어야 함:
#   import pydeck as pdk
#   import json

# -------------------------
# 0) 공통 유틸
# -------------------------
def _get_color_by_volume(val: int, max_val: int):
    """거래량이 많을수록 진한 빨강"""
    if max_val <= 0:
        return [255, 255, 200, 200]
    ratio = float(val) / float(max_val)
    ratio = max(0.0, min(1.0, ratio))
    g = int(255 * (1 - ratio))
    b = int(100 * (1 - ratio))
    return [255, g, b, 200]


# =========================================================
# 1) 법정동별 거래량 (ColumnLayer)
# =========================================================
# =========================================================
# ⚙️ 지도 시각화 설정 (본문 상단에 배치)
# =========================================================
st.title("💸 내 집 마련의 꿈 💸")

st.markdown("---")

st.subheader("📊 부동산 거래량 대시보드")
st.markdown("#### ⚙️ 지도 시각화 설정 ####")
view_option = st.radio(
    "보고 싶은 데이터를 선택하세요:",
    ("거래 금액 중앙값 (단위: 만원)", "평당 가격 중앙값 (단위: 만원)"),
    horizontal=True,
    key="map_view_option"
)

st.subheader("📍 법정동별 상세 거래량")
st.write("2010년, 2015년, 2020년, 2025년 지역별 아파트 거래량의 합")

@st.cache_data(show_spinner=True)
def load_dong_data_map(base_dir_str: str) -> pd.DataFrame:
    base_dir = Path(base_dir_str)
    path = base_dir / "법정동주소 거래량 데이터.csv"
    if not path.exists():
        raise FileNotFoundError(f"'{path.name}' 파일이 없습니다. 위치: {base_dir}")

    df = pd.read_csv(path)

    # 컬럼 후보(오타/변형 대응)
    col_lng = next((c for c in ["Longitude", "longitude", "lng", "LNG"] if c in df.columns), None)
    col_lat = next((c for c in ["Latitude", "latitude", "lat", "LAT"] if c in df.columns), None)
    col_name = next((c for c in ["법정동주소", "dong_name", "동", "법정동"] if c in df.columns), None)
    col_vol = next((c for c in ["지역별 거래량", "지역별 겨래량", "거래량", "volume", "VOL"] if c in df.columns), None)

    missing = [k for k, v in {"lng": col_lng, "lat": col_lat, "dong_name": col_name, "volume": col_vol}.items() if v is None]
    if missing:
        raise ValueError(
            "법정동 거래량 CSV 컬럼을 인식하지 못했습니다.\n"
            f"- 누락 키: {missing}\n"
            f"- 감지된 컬럼: {list(df.columns)}\n"
            "필요 예시: Longitude, Latitude, 법정동주소, 지역별 거래량"
        )

    df = df.rename(columns={col_lng: "lng", col_lat: "lat", col_name: "dong_name", col_vol: "volume"})

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

    df = df.dropna(subset=["lng", "lat"])
    return df


try:
    df_dong = load_dong_data_map(str(BASE_DIR))
    max_vol_dong = int(df_dong["volume"].max()) if not df_dong.empty else 0

    df_dong["bar_width"] = 0 if max_vol_dong == 0 else ((df_dong["volume"] / max_vol_dong) * 100).astype(int)
    df_dong["color"] = df_dong["volume"].apply(lambda x: _get_color_by_volume(int(x), max_vol_dong))

    tooltip_dong = {
        "html": """
            <div style="background: rgba(20, 20, 20, 0.95); padding: 12px; border-radius: 8px; color: white;
                        font-family: 'Segoe UI', sans-serif; box-shadow: 0 4px 6px rgba(0,0,0,0.3); min-width: 180px;">
                <div style="font-weight: bold; font-size: 1.1em; border-bottom: 1px solid #555;
                            margin-bottom: 8px; padding-bottom: 4px; color: #fff;">
                    📍 {dong_name}
                </div>

                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 0.9em; color: #ccc;">총 거래량</span>
                    <span style="font-weight: bold; font-size: 1.1em; color: #ff9f1c;">
                        {volume} <span style="font-size:0.7em; color:#aaa;">건</span>
                    </span>
                </div>

                <div style="width: 100%; background-color: #444; height: 10px; border-radius: 5px;
                            overflow: hidden; margin-bottom: 2px;">
                    <div style="width: {bar_width}%; background: linear-gradient(90deg, #ff9f1c, #ff5e00);
                                height: 100%;"></div>
                </div>

                <div style="text-align: right; font-size: 11px; color: #777;">
                    Max 대비 {bar_width}% 수준
                </div>
            </div>
        """,
        "style": {"color": "white"}
    }

    layer_dong = pdk.Layer(
        "ColumnLayer",
        data=df_dong,
        get_position=["lng", "lat"],
        get_elevation="volume",
        elevation_scale=7,
        radius=300,
        extruded=True,
        get_fill_color="color",
        pickable=True,
        auto_highlight=True,
    )

    view_state_dong = pdk.ViewState(
        longitude=127.5,
        latitude=36.0,
        zoom=6.5,
        pitch=45,
        bearing=0
    )

    r_dong = pdk.Deck(
        layers=[layer_dong],
        initial_view_state=view_state_dong,
        tooltip=tooltip_dong,
        map_style=pdk.map_styles.DARK
    )

    st.pydeck_chart(r_dong)

except Exception as e:
    st.error("❌ 법정동 거래량 지도 로딩 실패")
    st.exception(e)


# =========================================================
# 2) 광역자치단체 트렌드 (GeoJsonLayer)
#   - 'all' 파일 없이 자동 생성(2010/2015/2020/2025 평균)
# =========================================================
st.markdown("---")

def round_coordinates(coords, precision=4):
    if not coords:
        return coords
    if isinstance(coords[0], (int, float)):
        return [round(c, precision) for c in coords]
    return [round_coordinates(c, precision) for c in coords]


@st.cache_data(show_spinner=True)
def load_geo_data_map(base_dir_str: str) -> dict:
    base_dir = Path(base_dir_str)
    geo_path = base_dir / "대한민국_광역자치단체_경계 (1).geojson"
    if not geo_path.exists():
        raise FileNotFoundError(f"'{geo_path.name}' 파일이 없습니다. 위치: {base_dir}")

    with open(geo_path, encoding="utf-8") as f:
        geojson = json.load(f)

    # 좌표 단순화(성능 개선)
    for feature in geojson.get("features", []):
        geom = feature.get("geometry", {})
        if "coordinates" in geom:
            geom["coordinates"] = round_coordinates(geom["coordinates"])
    return geojson


@st.cache_data(show_spinner=True)
def load_apart_data_map(base_dir_str: str) -> dict:
    """
    apart_dict[year] = 시도별 데이터 (중복 제거)
    apart_dict['all'] = 2010/2015/2020/2025 평균으로 생성
    """
    base_dir = Path(base_dir_str)
    years = [2010, 2015, 2020, 2025]
    apart_dict = {}

    usecols = ["시도", "시도별_거래금액_중앙값", "시도별_평당가격_중앙값"]

    for y in years:
        path = base_dir / f"시도_apart_{y}_data.csv"
        if not path.exists():
            apart_dict[y] = pd.DataFrame()
            continue

        df = pd.read_csv(path, usecols=[c for c in usecols if c in pd.read_csv(path, nrows=0).columns])
        # 컬럼 유효성 체크
        if "시도" not in df.columns:
            apart_dict[y] = pd.DataFrame()
            continue

        # 후보 컬럼 보강(만약 중앙값 컬럼명이 다르면 여기서 추가 대응 가능)
        if "시도별_거래금액_중앙값" not in df.columns or "시도별_평당가격_중앙값" not in df.columns:
            apart_dict[y] = pd.DataFrame()
            continue

        df = df.drop_duplicates(subset=["시도"])
        # 숫자화
        df["시도별_거래금액_중앙값"] = pd.to_numeric(df["시도별_거래금액_중앙값"], errors="coerce")
        df["시도별_평당가격_중앙값"] = pd.to_numeric(df["시도별_평당가격_중앙값"], errors="coerce")
        apart_dict[y] = df

    # 'all' 생성: 연도별 df를 concat 후 시도별 평균(0 제외 평균은 원하면 바꿀 수 있음)
    frames = []
    for y in years:
        dfy = apart_dict.get(y, pd.DataFrame())
        if not dfy.empty:
            tmp = dfy[["시도", "시도별_거래금액_중앙값", "시도별_평당가격_중앙값"]].copy()
            tmp["year"] = y
            frames.append(tmp)

    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        # 시도별 평균
        all_df = all_df.groupby("시도", as_index=False)[["시도별_거래금액_중앙값", "시도별_평당가격_중앙값"]].mean()
        apart_dict["all"] = all_df
    else:
        apart_dict["all"] = pd.DataFrame()

    return apart_dict


@st.cache_data(show_spinner=True)
def process_map_data_map(geojson: dict, apart_dict: dict) -> pd.DataFrame:
    regions = [f["properties"]["CTP_KOR_NM"] for f in geojson.get("features", [])]
    df_map = pd.DataFrame({"시도": regions})

    years = [2010, 2015, 2020, 2025, "all"]
    for y in years:
        df_year = apart_dict.get(y, pd.DataFrame())
        if df_year is None or df_year.empty:
            continue

        temp = df_year[["시도", "시도별_거래금액_중앙값", "시도별_평당가격_중앙값"]].copy()
        temp = temp.rename(columns={
            "시도별_거래금액_중앙값": f"median_price_{y}",
            "시도별_평당가격_중앙값": f"pyeong_price_{y}",
        })
        df_map = pd.merge(df_map, temp, on="시도", how="left")

    # 결측은 0
    for c in df_map.columns:
        if c != "시도":
            df_map[c] = pd.to_numeric(df_map[c], errors="coerce").fillna(0).astype(int)
    return df_map


def get_fill_color_map(val: int, min_val: int, max_val: int):
    if val == 0:
        return [50, 50, 50, 150]
    if max_val <= min_val:
        return [100, 100, 100, 150]

    ratio = (val - min_val) / (max_val - min_val)
    ratio = max(0.0, min(1.0, ratio))
    r = 255
    g = int(255 * (1 - ratio))
    b = 0
    return [r, g, b, 200]


def generate_svg_chart_map(prices):
    width, height = 220, 80
    p_min, p_max = min(prices), max(prices)
    if p_min == p_max:
        p_max += 10

    def get_x(i): return 20 + (i / 3) * (width - 40)
    def get_y(p): return height - 20 - ((p - p_min) / (p_max - p_min) * (height - 40))

    points = " ".join([f"{get_x(i)},{get_y(p)}" for i, p in enumerate(prices)])

    elements = ""
    for i, p in enumerate(prices):
        cx, cy = get_x(i), get_y(p)
        elements += f'<circle cx="{cx}" cy="{cy}" r="3" fill="white" stroke="#d32f2f" stroke-width="2"/>'
        elements += f'<text x="{cx}" y="{cy-8}" fill="white" font-size="10" text-anchor="middle" font-weight="bold">{p}</text>'

    return (
        f'<svg width="{width}" height="{height}" style="background: rgba(0,0,0,0);">'
        f'<text x="{get_x(0)}" y="{height-5}" fill="#aaa" font-size="10" text-anchor="middle">2010</text>'
        f'<text x="{get_x(1)}" y="{height-5}" fill="#aaa" font-size="10" text-anchor="middle">2015</text>'
        f'<text x="{get_x(2)}" y="{height-5}" fill="#aaa" font-size="10" text-anchor="middle">2020</text>'
        f'<text x="{get_x(3)}" y="{height-5}" fill="#aaa" font-size="10" text-anchor="middle">2025</text>'
        f'<polyline points="{points}" fill="none" stroke="#d32f2f" stroke-width="2"/>'
        f'{elements}</svg>'
    )


@st.cache_data(show_spinner=True)
def precompute_visual_assets_map(base_dir_str: str):
    apart_dict = load_apart_data_map(base_dir_str)
    geojson_data = load_geo_data_map(base_dir_str)

    if not geojson_data.get("features"):
        return None, {}

    df_map = process_map_data_map(geojson_data, apart_dict)
    price_dict = df_map.set_index("시도").to_dict("index")

    stats = {}
    for prefix in ["median_price", "pyeong_price"]:
        col = f"{prefix}_all"
        vals = df_map[df_map[col] > 0][col]
        stats[prefix] = (int(vals.min()), int(vals.max())) if not vals.empty else (0, 100)

    assets_cache = {}
    for region_name, row in price_dict.items():
        assets_cache[region_name] = {}
        for prefix in ["median_price", "pyeong_price"]:
            p_prices = [int(row.get(f"{prefix}_{y}", 0)) for y in [2010, 2015, 2020, 2025]]
            p_val = int(row.get(f"{prefix}_all", 0))
            p_min, p_max = stats[prefix]

            assets_cache[region_name][prefix] = {
                "value": p_val,
                "color": get_fill_color_map(p_val, p_min, p_max),
                "chart": generate_svg_chart_map(p_prices)
            }

    return geojson_data, assets_cache


if "거래 금액 중앙값" in view_option:
    target_prefix = "median_price"
    chart_title = "거래 금액 중앙값"
else:
    target_prefix = "pyeong_price"
    chart_title = "평당 가격 중앙값"

st.subheader(f"📉 17개 시도 아파트 {chart_title} 트렌드 (2010, 2015, 2020, 2025)")
st.markdown("지도 색상은 **4개년 평균(all)** 기준이며, 툴팁은 **연도별 변화**를 보여줍니다.")

try:
    geojson_data, assets_cache = precompute_visual_assets_map(str(BASE_DIR))

    if geojson_data and assets_cache:
        for feature in geojson_data["features"]:
            region_name = feature["properties"]["CTP_KOR_NM"]

            if region_name in assets_cache:
                data = assets_cache[region_name][target_prefix]
                feature["properties"]["current_value"] = data["value"]
                feature["properties"]["fill_color"] = data["color"]
                feature["properties"]["svg_chart"] = data["chart"]
                feature["properties"]["chart_title"] = chart_title
            else:
                feature["properties"]["current_value"] = 0
                feature["properties"]["fill_color"] = [50, 50, 50, 150]
                feature["properties"]["svg_chart"] = ""

        layer = pdk.Layer(
            "GeoJsonLayer",
            geojson_data,
            pickable=True,
            stroked=True,
            filled=True,
            extruded=False,
            get_fill_color="properties.fill_color",
            get_line_color=[255, 255, 255, 100],
            line_width_min_pixels=1,
            auto_highlight=True,
        )

        tooltip = {
            "html": """
            <div style="background: rgba(0, 0, 0, 0.85); padding: 15px; border-radius: 10px; color: white;
                        font-family: sans-serif; box-shadow: 0 4px 6px rgba(0,0,0,0.3);">
                <div style="font-weight: bold; font-size: 16px; margin-bottom: 5px; border-bottom: 1px solid #555; padding-bottom: 5px;">
                    📍 {CTP_KOR_NM}
                </div>
                <div style="font-size: 12px; color: #ccc; margin-bottom: 10px;">
                    4개년 평균(all) 기준:
                    <span style="color: #ffeb3b; font-weight: bold; font-size: 14px;">{current_value}</span> 만원
                </div>
                <div style="background: rgba(255,255,255,0.05); border-radius: 5px; padding: 5px;">
                    {svg_chart}
                </div>
            </div>
            """,
            "style": {"color": "white"}
        }

        view_state = pdk.ViewState(longitude=127.5, latitude=36.0, zoom=6, pitch=0)

        r = pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style=pdk.map_styles.DARK
        )

        st.pydeck_chart(r, width="stretch", height=700)

    else:
        st.error("지도 데이터를 불러오지 못했습니다. (geojson 또는 자산 캐시가 비어 있음)")

except Exception as e:
    st.error("❌ 광역시도 지도 로딩 실패")
    st.exception(e)



st.markdown("---")


# =========================
# 2) 타이틀
# =========================
st.subheader("📌 아파트 가격 상승 추세")

# =========================
# 3) 스타일(요청 반영: 블랙+블루 / 선 두께)
# =========================
BLACK = "#444444"
BLUE = "#74a7fe"
LINE_W_THICK = 4   # 구매가능/개월 그래프 선 두께
LINE_W_NORMAL = 4  # 지수/덤벨 선 두께

# =========================
# 4) 유틸
# =========================
def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def make_index(series: pd.Series, years: pd.Series, base_year: int = 2010) -> pd.Series:
    mask = years == base_year
    if mask.sum() == 0:
        raise ValueError(f"기준연도({base_year})가 데이터에 없습니다. 가능한 연도: {sorted(years.unique().tolist())}")
    base = series[mask].iloc[0]
    return (series / base) * 100.0

# =========================
# 5) 로딩 함수 (캐시는 "위젯 없는" 순수 함수만)
# =========================
@st.cache_data(show_spinner=True)
def load_wage_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

def prepare_wage_df(wage_raw: pd.DataFrame, item_choice: str, hh_choice: str) -> pd.DataFrame:
    year_col = "Year" if "Year" in wage_raw.columns else ("year" if "year" in wage_raw.columns else None)
    if year_col is None:
        raise ValueError(f"소득 데이터에 Year/year 컬럼이 없습니다. 현재 컬럼: {list(wage_raw.columns)}")

    required = {"item", "hh", "value"}
    if not required.issubset(set(wage_raw.columns)):
        raise ValueError(f"소득 데이터에 item/hh/value 컬럼이 없습니다. 현재 컬럼: {list(wage_raw.columns)}")

    out = wage_raw.copy()
    out = out[out["item"] == item_choice]
    out = out[out["hh"] == hh_choice]

    out[year_col] = pd.to_numeric(out[year_col], errors="coerce")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=[year_col, "value"])

    out = out.rename(columns={year_col: "year", "value": "income_value"})
    out["year"] = out["year"].astype(int)

    out = out.groupby("year", as_index=False)["income_value"].mean()
    return out.sort_values("year").reset_index(drop=True)

@st.cache_data(show_spinner=True)
def load_apart_auto(apart_paths: list[str]) -> pd.DataFrame:
    """
    전국 대표 평당가격 = (시도별 평당가격) 을 (시도별 총 거래금액)으로 가중평균
    - price_col 우선: 시도별_평당가격_중앙값, 그 다음 시도별_평균_평당가격
    - weight_col 우선: 시도별_총_거래금액
    """
    rows = []
    for p in sorted(apart_paths):
        name = nfc(Path(p).name)
        m = re.search(r"^시도_apart_(\d{4})_data\.csv$", name)
        if not m:
            continue
        year = int(m.group(1))

        df = pd.read_csv(p)

        # ✅ 가격 컬럼 후보(너 데이터 기준)
        price_col = pick_col(df, [
            "시도별_평당가격_중앙값",
            "시도별_평균_평당가격",
            "시도별_평당가격중앙값",
            "시도별_평균평당가격",
        ])

        # ✅ 가중치(거래금액) 컬럼 후보
        weight_col = pick_col(df, [
            "시도별_총_거래금액",
            "시도별_총거래금액",
            "총_거래금액",
            "총거래금액",
        ])

        if price_col is None:
            raise ValueError(
                f"{Path(p).name} 에서 '평당가격' 컬럼을 찾지 못했습니다.\n"
                f"- 감지된 컬럼: {list(df.columns)}\n"
                "필요 후보: 시도별_평당가격_중앙값 / 시도별_평균_평당가격"
            )

        if weight_col is None:
            raise ValueError(
                f"{Path(p).name} 에서 '총 거래금액(가중치)' 컬럼을 찾지 못했습니다.\n"
                f"- 감지된 컬럼: {list(df.columns)}\n"
                "필요 후보: 시도별_총_거래금액"
            )

        price = pd.to_numeric(df[price_col], errors="coerce")
        w = pd.to_numeric(df[weight_col], errors="coerce")

        tmp = pd.DataFrame({"price": price, "w": w}).dropna()
        tmp = tmp[tmp["w"] > 0]

        if tmp.empty:
            raise ValueError(
                f"{Path(p).name}: 가중평균 계산용 데이터가 비어있습니다.\n"
                f"price_col={price_col}, weight_col={weight_col}"
            )

        # ✅ 거래금액 가중평균
        weighted_avg = float((tmp["price"] * tmp["w"]).sum() / tmp["w"].sum())

        rows.append({
            "year": year,
            "apt_price_median": weighted_avg,          # (이름은 그대로 쓰되 의미는 가중평균)
            "apt_price_col_used": price_col,
            "apt_weight_col_used": weight_col
        })

    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    if out.empty:
        raise ValueError("아파트 파일에서 연도별 집계를 만들지 못했습니다.")
    return out

# =========================
# 6) 소득 데이터: 위젯 선택 (캐시 밖)
# =========================
try:
    wage_raw = load_wage_raw(WAGE_PATH)
except Exception as e:
    st.error("❌ 소득 파일 로딩 실패")
    st.exception(e)
    st.stop()

items = sorted(wage_raw["item"].dropna().unique())
hhs = sorted(wage_raw["hh"].dropna().unique())

default_hh_idx = hhs.index("1인") if "1인" in hhs else 0
default_item_idx = 0

col1, col2 = st.columns(2)
with col1:
    item_choice = st.selectbox("소득 항목 선택", items, index=default_item_idx)
with col2:
    hh_choice = st.selectbox("가구유형 선택", hhs, index=default_hh_idx)

try:
    wage_df = prepare_wage_df(wage_raw, item_choice=item_choice, hh_choice=hh_choice)
except Exception as e:
    st.error("❌ 소득 데이터 처리 실패")
    st.exception(e)
    st.stop()

# =========================
# 7) 아파트 데이터 로딩
# =========================
try:
    apt_df = load_apart_auto(apart_files)
except Exception as e:
    st.error("❌ 아파트 데이터 로딩 실패")
    st.exception(e)
    st.stop()

# =========================
# 8) 병합 + 지수화
# =========================
merged = pd.merge(wage_df, apt_df, on="year", how="inner").sort_values("year").reset_index(drop=True)
if merged.empty:
    st.error("❌ 소득 연도와 아파트 연도가 겹치지 않습니다.")
    st.write("소득 연도:", wage_df["year"].tolist())
    st.write("아파트 연도:", apt_df["year"].tolist())
    st.stop()

try:
    merged["Income_Index"] = make_index(merged["income_value"], merged["year"], base_year=2010)
    merged["Apartment_Index"] = make_index(merged["apt_price_median"], merged["year"], base_year=2010)
except Exception as e:
    st.error("❌ 지수화(2010=100) 실패")
    st.exception(e)
    st.stop()

df = merged.rename(columns={"year": "Year"}).copy()

latest_year = int(df["Year"].iloc[-1])
gap_2010 = float(df.loc[df["Year"] == 2010, "Apartment_Index"].iloc[0] - df.loc[df["Year"] == 2010, "Income_Index"].iloc[0])
gap_latest = float(df["Apartment_Index"].iloc[-1] - df["Income_Index"].iloc[-1])

st.success(
    f"✅ 병합 완료: {df['Year'].min()} ~ {df['Year'].max()} (기준연도=2010=100)\n\n"
    f"- 소득 선택: {item_choice}, 가구원 수: {hh_choice}\n"
    f"- 아파트 가격 컬럼: {', '.join(df['apt_price_col_used'].unique())}\n"
    f"- 아파트 대표값: 시도별 평당가격을 '총 거래금액'으로 가중평균(전국 대표)"
)

st.markdown(
    f"""
- 2010년 기준 격차(아파트-소득): **{gap_2010:.1f}p**
- {latest_year}년 격차(아파트-소득): **{gap_latest:.1f}p**
"""
)

st.divider()

# =========================
# 9) (1) 지수 영역 차트 (블랙 + 블루)
# =========================
st.subheader(" 📈 소득 vs 아파트 가격 지수 (2010=100)")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["Year"], y=df["Income_Index"],
    mode="lines+markers",
    name="소득 지수",
    line=dict(color=BLUE, width=LINE_W_NORMAL),
    marker=dict(size=9, color=BLUE),
    fill="tozeroy",
    fillcolor="rgba(31,119,180,0.10)"
))
fig.add_trace(go.Scatter(
    x=df["Year"], y=df["Apartment_Index"],
    mode="lines+markers",
    name="아파트 가격 지수",
    line=dict(color=BLACK, width=LINE_W_NORMAL),
    marker=dict(size=9, color=BLACK),
    fill="tonexty",
    fillcolor="rgba(17,17,17,0.10)"
))
fig.update_layout(
    title="연도별 소득 vs 아파트 가격 지수 (2010=100)",
    xaxis_title="연도",
    yaxis_title="지수",
    hovermode="x unified",
    height=520
)
st.plotly_chart(fig, use_container_width=True)

st.info(
    " **지수는 2010년을 기준연도로 설정하여 2025년까지의 변화를 나타냄** "
)

st.divider()

# =========================
# 10) (2) 덤벨 차트 (블랙 + 블루)
# =========================
st.subheader(" 🏋️ 덤벨 차트: 연도별 격차(아파트-소득)")

ddf = df.copy()
ddf["Year_str"] = ddf["Year"].astype(str)

fig2 = go.Figure()

# 연결선(덤벨 바) - 블랙
for i in range(len(ddf)):
    fig2.add_shape(
        type="line",
        x0=float(ddf.loc[i, "Income_Index"]), y0=ddf.loc[i, "Year_str"],
        x1=float(ddf.loc[i, "Apartment_Index"]), y1=ddf.loc[i, "Year_str"],
        line=dict(color=BLACK, width=6)
    )

# 소득 점(블루)
fig2.add_trace(go.Scatter(
    x=ddf["Income_Index"], y=ddf["Year_str"],
    mode="markers",
    name="소득 지수",
    marker=dict(size=12, color=BLUE),
    hovertemplate="연도: %{y}<br>소득 지수: %{x:.1f}<extra></extra>"
))

# 아파트 점(블랙)
fig2.add_trace(go.Scatter(
    x=ddf["Apartment_Index"], y=ddf["Year_str"],
    mode="markers",
    name="아파트 가격 지수",
    marker=dict(size=12, color=BLACK),
    hovertemplate="연도: %{y}<br>아파트 지수: %{x:.1f}<extra></extra>"
))

fig2.update_layout(
    title="연도별 소득 vs 아파트 가격 지수 격차 (덤벨)",
    xaxis_title="지수(2010=100)",
    yaxis_title="연도",
    height=560,
    margin=dict(l=90, r=40, t=90, b=50),
    hovermode="closest",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)
st.plotly_chart(fig2, use_container_width=True)

st.info(f" **격차(지수): 2010년 {gap_2010:.1f} → {latest_year}년 {gap_latest:.1f}**")

st.divider()

# =========================
# 11) (3) 구매가능 면적(평) + 1평 구매에 필요한 '개월' 설명 정리
# =========================
st.subheader(" 📏 구매가능 면적(평) & 1평 구매에 필요한 기간(개월)")


# 사용자가 소득 주기를 선택하도록 해서 '개월' 의미를 확정
# ✅ 입력 value는 "월소득(원)"으로 고정 (중복 환산 방지)
st.caption("소득 데이터(value)는 '월소득(원)', 아파트 가격은 '만원/평'기준입니다.")
df["income_monthly_won"] = df["income_value"]


# 아파트 단위 정규화 → 원/평
# ✅ 아파트 평당가격은 '만원/평'으로 고정 → 원/평으로 변환
df["apt_price_per_pyeong_won"] = df["apt_price_median"] * 10000

# 구매가능 평수(월소득 기준)
df["Purchasable_Pyeong"] = df["income_monthly_won"] / df["apt_price_per_pyeong_won"]

# ✅ 1평 구매에 필요한 개월 수(월소득 기준) — 이게 진짜 "개월"임
df["Months_for_1Pyeong"] = df["apt_price_per_pyeong_won"] / df["income_monthly_won"]

st.caption(
    "정의: 1평 구매 개월 수 = (평당가격 만원/평) ÷ (월소득 원/월). "
)

# (3-1) 구매가능 평수 그래프: 블랙 + 두꺼운 선
fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=df["Year"], y=df["Purchasable_Pyeong"],
    mode="lines+markers",
    name="구매가능 평수(평)",
    line=dict(color=BLACK, width=LINE_W_THICK),
    marker=dict(size=8, color=BLACK)
))
fig3.update_layout(
    title="연도별 구매가능 평수(월소득/평당가격)",
    xaxis_title="연도",
    yaxis_title="평",
    hovermode="x unified",
    height=460
)
st.plotly_chart(fig3, use_container_width=True)

# (3-2) 1평 구매 개월 수 그래프: 블랙 + 동일 두께
fig4 = go.Figure()
fig4.add_trace(go.Scatter(
    x=df["Year"], y=df["Months_for_1Pyeong"],
    mode="lines+markers",
    name="1평 구매에 필요한 개월 수",
    line=dict(color=BLACK, width=LINE_W_THICK),
    marker=dict(size=8, color=BLACK)
))
fig4.update_layout(
    title="연도별 1평 구매에 필요한 기간(개월) = 평당가격/월소득",
    xaxis_title="연도",
    yaxis_title="개월",
    hovermode="x unified",
    height=460
)
st.plotly_chart(fig4, use_container_width=True)


#section 2 
# =========================================================
# 📌 페르소나 섹션 (지수화 분석 이후)
# =========================================================
st.markdown("---")
st.subheader("🏠 내 집 마련의 꿈 🏠")

st.markdown("**이O현: 내 집은 어디에 있을까?**")

image_path = BASE_DIR / "image.png"

if image_path.exists():
    image = Image.open(image_path)
    st.image(
        image,
        caption="이O현님의 내 집을 찾아줘",
        use_container_width=True
    )
else:
    st.error(f"이미지 파일을 찾을 수 없습니다: {image_path.name}")
    st.info(f"'{image_path.name}' 파일을 이 스크립트와 같은 폴더에 넣어주세요.")

st.markdown("---")

# 📋 프로필 카드
st.subheader("📋 이O현 개인프로필 상세")

col1, col2 = st.columns(2)

with col1:
    st.write("**이름:** 이O현")
    st.write("**나이:** 29세 (만 27세)")
    st.write("**직업:** 평범하고 성실한 직장인")

with col2:
    st.write("**현재 상태:** 미혼 (결혼 예정)")
    st.write("**목표 주택:** 서울/수도권 24평 아파트")
    st.write("**긴급도:** 🔥 매우 높음, 최대한 빨리 결혼하고 싶음")  

st.markdown("---")

# 💭 시나리오 설명
st.subheader("💭 내 집 마련의 꿈 시나리오 배경")
st.markdown("""
> **"수 많은 주택 데이터가 나에게는 어떤 의미를 가질까요?"**

앞에서 본 **소득–아파트 가격 격차의 구조적 문제**를  
이제 한 명의 현실적인 인물에게 적용해봅니다.

이 시뮬레이션은  
**소득 성장률**, **저축 전략**, **결혼(맞벌이)** 이  
내 집 마련 가능 시점을 어떻게 바꾸는지를 보여주기 위한 출발점입니다.
""")

#========은정_시뮬레이터=============
# =========================================================
# 🧮 (추가) 내 집 마련 통합 시뮬레이터 섹션
#   ✅ 위 코드 내용은 그대로 두고, import/set_page_config/타이틀 중복 없이
#   ✅ BASE_DIR + 이미 존재하는 데이터(소득파일/아파트파일) 기준으로만 연결
# =========================================================
st.markdown("---")
st.header("🏠 내 집 마련 시뮬레이터")
st.markdown("#### 📉 아파트 실거래가와 소득 데이터를 적용하여 시뮬레이션")

# --------------------------------------------------------------------------
# 1) 데이터 로드 및 병합 (프로젝트 데이터 구조 반영)
#    - 소득: 1인당_개인소득.csv (Year/item/hh/value)
#    - 아파트: 시도_apart_YYYY_data.csv (시도별_거래금액_중앙값 사용)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_and_merge_data_simulator(base_dir_str: str):
    base_dir = Path(base_dir_str)

    # ✅ 소득 파일(프로젝트)
    wage_path = base_dir / "1인당_개인소득.csv"
    if not wage_path.exists():
        return None, f"'{wage_path.name}' 파일이 없습니다."

    wage_df = pd.read_csv(wage_path)
    if "Unnamed: 0" in wage_df.columns:
        wage_df = wage_df.drop(columns=["Unnamed: 0"])

    year_col = "Year" if "Year" in wage_df.columns else ("year" if "year" in wage_df.columns else None)
    required = {"item", "hh", "value"}
    if year_col is None or not required.issubset(set(wage_df.columns)):
        return None, f"소득 파일 컬럼이 예상과 다릅니다. 감지된 컬럼: {list(wage_df.columns)}"

    # ✅ 내용 변경 없이 자동 선택(기존과 동일): item=처분가능소득 우선, hh=1인 우선
    items = wage_df["item"].dropna().unique().tolist()
    hhs = wage_df["hh"].dropna().unique().tolist()
    item_choice = "처분가능소득" if "처분가능소득" in items else (items[0] if items else None)
    hh_choice = "1인" if "1인" in hhs else (hhs[0] if hhs else None)

    if item_choice is None or hh_choice is None:
        return None, "소득 파일에서 item/hh 값을 읽을 수 없습니다."

    wage_selected = wage_df.loc[
        (wage_df["item"] == item_choice) & (wage_df["hh"] == hh_choice),
        [year_col, "value"]
    ].copy()

    wage_selected[year_col] = pd.to_numeric(wage_selected[year_col], errors="coerce")
    wage_selected["value"] = pd.to_numeric(wage_selected["value"], errors="coerce")
    wage_selected = wage_selected.dropna()

    # ✅ value는 '월소득(원)' → 연소득으로 환산해서 연봉 기반 저축에 사용
    wage_selected = wage_selected.rename(columns={year_col: "year", "value": "monthly_wage_won"})
    wage_selected["year"] = wage_selected["year"].astype(int)

    # 연도별 평균 월소득(원)
    wage_selected = wage_selected.groupby("year", as_index=False)["monthly_wage_won"].mean()

    # 연소득(원) = 월소득 × 12
    wage_selected["annual_wage_won"] = wage_selected["monthly_wage_won"] * 12

    # 연소득(만원)
    wage_selected["annual_wage_manwon"] = wage_selected["annual_wage_won"] / 10000.0

    # ✅ 아파트 파일(프로젝트) - BASE_DIR 기준
    files_in_dir_local = sorted([p.name for p in base_dir.iterdir()])
    apt_pat_local = re.compile(r"^시도_apart_(\d{4})_data\.csv$")
    apart_paths = sorted([base_dir / fn for fn in files_in_dir_local if apt_pat_local.match(nfc(fn))])

    if not apart_paths:
        return None, "아파트 데이터 파일(시도_apart_YYYY_data.csv)을 찾을 수 없습니다."

    apart_data_list = []
    for path in apart_paths:
        m = re.search(r"시도_apart_(\d{4})_data\.csv$", nfc(path.name))
        if not m:
            continue
        year = int(m.group(1))

        temp_df = pd.read_csv(path)

        # ✅ 시뮬레이터는 거래금액(만원)을 써야 기존 계산/표현이 그대로 유지됨
        if "시도별_거래금액_중앙값" in temp_df.columns and "시도" in temp_df.columns:
            sub_df = temp_df[["시도", "시도별_거래금액_중앙값"]].copy()
            sub_df["year"] = year
            apart_data_list.append(sub_df)

    if not apart_data_list:
        return None, "아파트 파일은 있지만 '시도별_거래금액_중앙값' 컬럼을 가진 데이터를 찾지 못했습니다."

    apart_all = pd.concat(apart_data_list, ignore_index=True)

    # 병합(기존과 동일)
    merged_df = pd.merge(apart_all, wage_selected[["year", "annual_wage_manwon"]], on="year", how="left")
    return merged_df, None


raw_data_sim, error_message_sim = load_and_merge_data_simulator(str(BASE_DIR))


# --------------------------------------------------------------------------
# 2) UI (시뮬레이션 설정 + 기준연도 통합)
# --------------------------------------------------------------------------
with st.container():
    st.subheader("⚙️ 시뮬레이션 설정")

    col0, col1, col2, col3 = st.columns(4)

    with col0:
        available_years = sorted(raw_data_sim["year"].unique())
        default_idx = len(available_years) - 1

        selected_year = st.selectbox(
            "🗓️ 데이터 기준 연도",
            available_years,
            index=default_idx,
            help="이 연도의 소득·아파트 데이터를 기준으로 시뮬레이션합니다.",
            key="sim_selected_year"
        )

    with col1:
        savings_rate = st.slider(
            "💰 저축률 (%)",
            min_value=10, max_value=100, value=50, step=5,
            key="sim_savings_rate"
        )

    with col2:
        salary_growth_rate = st.slider(
            "📈 매년 연봉 상승률 (%)",
            min_value=0.0, max_value=10.0, value=3.0, step=0.5,
            key="sim_salary_growth_rate"
        )

    with col3:
    # ✅ 배우자 기본값 = 본인 연봉과 동일
        subset_for_default = raw_data_sim[raw_data_sim["year"] == selected_year]
        my_income = subset_for_default["annual_wage_manwon"].dropna().mean()
        default_spouse = int(my_income) if pd.notna(my_income) else 4000

        spouse_income = st.slider(
             "👫 배우자 연봉 (만원)",
            min_value=0,
            max_value=10000,
            value=default_spouse,
            step=100,
            help="기본값은 본인 연봉과 동일하게 설정됩니다.",
            key="sim_spouse_income"
    )

st.info(f"✅ {selected_year}년 데이터를 기준으로 시뮬레이션합니다.")


    # --------------------------------------------------------------------------
    # 4) 계산 로직 (기존 그대로)
    # --------------------------------------------------------------------------
def calculate_years(target_price, initial_income, save_rate, growth_rate):
    saved_amount = 0
    years = 0
    current_income = initial_income

    if initial_income <= 0 or pd.isna(target_price):
        return 999

    while saved_amount < target_price and years < 100:
        annual_saving = current_income * (save_rate / 100)
        saved_amount += annual_saving
        current_income *= (1 + growth_rate / 100)
        years += 1

    return years

if raw_data_sim is None:
    st.error("🚨 데이터 파일을 찾을 수 없습니다.")
    st.warning(f"오류 내용: {error_message_sim}")
    st.markdown("""
**해결 방법:**
1. 다음 파일들이 이 파이썬 파일과 같은 폴더에 있는지 확인해주세요.
   - `1인당_개인소득.csv`
   - `시도_apart_2010_data.csv`
   - `시도_apart_2015_data.csv`
   - `시도_apart_2020_data.csv`
   - `시도_apart_2025_data.csv`
   - (아파트 파일에는 `시도별_거래금액_중앙값` 컬럼이 있어야 합니다.)
""")
    st.stop()

    # --------------------------------------------------------------------------
# 5) 결과 탭 (기준연도 설정 탭 제거: 결과만 3개)
#   - tab1: 저축의 힘(기본)
#   - tab2: 성장의 힘(연봉상승)
#   - tab3: 함께의 힘(맞벌이)
# --------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "📊 저축의 힘 (기본)",
    "📈 성장의 힘 (연봉상승)",
    "👫 함께의 힘 (맞벌이)"
])

# ✅ 위쪽 시뮬레이션 설정에서 selected_year를 이미 선택했다고 가정
subset = raw_data_sim[raw_data_sim["year"] == selected_year].copy()

df_sim = pd.DataFrame({
    "지역": subset["시도"],
    "본인연봉(중위)": subset["annual_wage_manwon"],        # 만원
    "아파트중위가격": subset["시도별_거래금액_중앙값"],    # 만원
}).dropna(subset=["본인연봉(중위)", "아파트중위가격"])

df_sim = df_sim.sort_values("아파트중위가격", ascending=False).reset_index(drop=True)

# =========================
# TAB 1) 저축의 힘 (기본)
# =========================


with tab1:
    df_basic = df_sim.copy()

    df_basic["본인연봉(중위)"] = pd.to_numeric(df_basic["본인연봉(중위)"], errors="coerce")
    df_basic["아파트중위가격"] = pd.to_numeric(df_basic["아파트중위가격"], errors="coerce")
    df_basic = df_basic.dropna(subset=["본인연봉(중위)", "아파트중위가격"])

    if df_basic.empty:
        st.error("그래프를 그릴 데이터가 없습니다. (연봉/아파트가격 결측)")
        st.stop()

    df_basic["소요시간"] = df_basic.apply(
        lambda x: calculate_years(x["아파트중위가격"], x["본인연봉(중위)"], savings_rate, 0),
        axis=1
    )

    df_basic["소요시간"] = pd.to_numeric(df_basic["소요시간"], errors="coerce")
    df_basic = df_basic.dropna(subset=["소요시간"])

    df_basic["예상구매연도"] = selected_year + df_basic["소요시간"]  # base_year 있으면 base_year로 바꾸는 게 더 좋음
    df_basic = df_basic.sort_values("소요시간", ascending=False)

    fig1 = px.bar(
        df_basic,
        x="지역",
        y="소요시간",
        color="소요시간",
        text="소요시간",
        title="지역별 내 집 마련 소요 시간 (년)",
        hover_data={"예상구매연도": True},
        color_continuous_scale="Blues"
    )
    fig1.update_traces(texttemplate="%{text}년", textposition="outside", cliponaxis=False)

    st.plotly_chart(fig1, use_container_width=True)


# =========================
# TAB 2) 성장의 힘 (연봉상승)
# =========================
with tab2:
    df_growth = df_sim.copy()
    df_growth["고정연봉"] = df_growth.apply(
        lambda x: calculate_years(x["아파트중위가격"], x["본인연봉(중위)"], savings_rate, 0),
        axis=1
    )
    df_growth["상승연봉"] = df_growth.apply(
        lambda x: calculate_years(x["아파트중위가격"], x["본인연봉(중위)"], savings_rate, salary_growth_rate),
        axis=1
    )

    df_growth["고정_예상구매연도"] = selected_year + df_growth["고정연봉"]
    df_growth["상승_예상구매연도"] = selected_year + df_growth["상승연봉"]

    df_melted = df_growth.melt(
        id_vars="지역",
        value_vars=["고정연봉", "상승연봉"],
        var_name="구분",
        value_name="소요시간"
    )
    df_melted["구분"] = df_melted["구분"].map({
        "고정연봉": "❌ 연봉 동결",
        "상승연봉": f"⭕ 매년 {salary_growth_rate}% 상승"
    })

    st.subheader(f"📈 연봉 상승률 {salary_growth_rate}% 적용 효과 (기준={selected_year}년)")

    fig2 = px.bar(
        df_melted, x="지역", y="소요시간",
        color="구분", barmode="group", text="소요시간",
        title="연봉 상승 유무 비교"
    )
    fig2.update_traces(texttemplate="%{text}년", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

# =========================
# TAB 3) 함께의 힘 (맞벌이)
# =========================
with tab3:
    df_mate = df_sim.copy()
    df_mate["외벌이"] = df_mate.apply(
        lambda x: calculate_years(x["아파트중위가격"], x["본인연봉(중위)"], savings_rate, salary_growth_rate),
        axis=1
    )
    df_mate["맞벌이"] = df_mate.apply(
        lambda x: calculate_years(x["아파트중위가격"], x["본인연봉(중위)"] + spouse_income, savings_rate, salary_growth_rate),
        axis=1
    )

    df_mate["외벌이_예상구매연도"] = selected_year + df_mate["외벌이"]
    df_mate["맞벌이_예상구매연도"] = selected_year + df_mate["맞벌이"]

    df_melted_mate = df_mate.melt(
        id_vars="지역",
        value_vars=["외벌이", "맞벌이"],
        var_name="구분",
        value_name="소요시간"
    )
    df_melted_mate["구분"] = df_melted_mate["구분"].map({"외벌이": "🧍 외벌이", "맞벌이": "👫 맞벌이"})

    st.subheader(f"👫 배우자 연봉 {spouse_income:,}만원 합산 효과 (기준={selected_year}년)")

    fig3 = px.bar(
        df_melted_mate, x="지역", y="소요시간",
        color="구분", barmode="group", text="소요시간",
        title="외벌이 vs 맞벌이 비교"
    )
    fig3.update_traces(texttemplate="%{text}년", textposition="outside", cliponaxis=False)
    st.plotly_chart(fig3, use_container_width=True)

#==========시뮬레이터 끝=================

st.header("🏃‍♂️ 달려도 잡을 수 없는 집 — 시간 시뮬레이션")

region = st.radio("지역 선택", ["서울", "경기", "지방"], horizontal=True)

INCOME_PATH = BASE_DIR / "1인당_개인소득.csv"

APT_COL = "시도별_거래금액_중앙값"
base_years = [2010, 2015, 2020, 2025]

scenarios = [
    "1인·가처분소득",
    "캥거루·근로소득",
    "맞벌이·소득2배",
    "+주식·소득3배",   # ✅ 3.5배 → 3배로 표기 변경
]
lane_y = {scenarios[0]: 3, scenarios[1]: 2, scenarios[2]: 1, scenarios[3]: 0}

@st.cache_data(show_spinner=False)
def load_income_raw() -> pd.DataFrame:
    df = pd.read_csv(INCOME_PATH)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    if "Year" in df.columns and "year" not in df.columns:
        df = df.rename(columns={"Year": "year"})

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["year", "item", "hh", "value"])
    df["year"] = df["year"].astype(int)
    return df

@st.cache_data(show_spinner=False)
def get_monthly_income_won(
    year: int,
    item_name: str = "처분가능소득",
    hh_name: str = "1인",
) -> float:
    df = load_income_raw()
    sub = df[(df["year"] == year) & (df["hh"] == hh_name)]
    hit = sub[sub["item"] == item_name]
    if hit.empty:
        hit = sub[sub["item"].astype(str).str.contains(item_name, na=False)]
    if hit.empty:
        raise ValueError(f"{year}년 소득 데이터에서 item={item_name}, hh={hh_name}를 찾지 못했습니다.")
    return float(hit["value"].mean())

@st.cache_data(show_spinner=False)
def get_house_price_manwon(year: int, region_label: str) -> float:
    path = BASE_DIR / f"시도_apart_{year}_data.csv"
    apt = pd.read_csv(path)

    if APT_COL not in apt.columns:
        raise ValueError(f"{path.name} 에 '{APT_COL}' 컬럼이 없습니다. 현재 컬럼: {list(apt.columns)}")

    apt[APT_COL] = pd.to_numeric(apt[APT_COL], errors="coerce")
    apt = apt.dropna(subset=[APT_COL])

    if region_label == "서울":
        row = apt[apt["시도"].astype(str).str.contains("서울")]
        if row.empty:
            raise ValueError(f"{path.name}: '서울' 시도 행을 찾지 못했습니다.")
        return float(row[APT_COL].iloc[0])

    if region_label == "경기":
        row = apt[apt["시도"].astype(str).str.contains("경기")]
        if row.empty:
            raise ValueError(f"{path.name}: '경기' 시도 행을 찾지 못했습니다.")
        return float(row[APT_COL].iloc[0])

    other = apt[~apt["시도"].astype(str).str.contains("서울|경기")]
    if other.empty:
        raise ValueError(f"{path.name}: 지방(서울/경기 제외) 데이터가 비어있습니다.")
    return float(other[APT_COL].median())

@st.cache_data(show_spinner=False)
def build_data_real() -> dict:
    SAVE_RATE = 0.50
    KANGAROO_SAVE_RATE = 1.00
    COUPLE_MULT = 2.0

    # ✅ 주식 가정 세분화: "총소득 = 기존소득 × 3"
    STOCK_TOTAL_INCOME_MULT = 3.0

    out = {reg: {s: {} for s in scenarios} for reg in ["서울", "경기", "지방"]}

    for reg in ["서울", "경기", "지방"]:
        for y in base_years:
            house_price_manwon = get_house_price_manwon(y, reg)

            monthly_income_won = get_monthly_income_won(y, item_name="처분가능소득", hh_name="1인")
            annual_income_manwon = (monthly_income_won * 12.0) / 10000.0

            save_1 = annual_income_manwon * SAVE_RATE
            save_2 = annual_income_manwon * KANGAROO_SAVE_RATE
            save_3 = (annual_income_manwon * COUPLE_MULT) * SAVE_RATE

            # ✅ 총소득을 3배로 만든 뒤 저축률(50%) 적용
            save_4 = (annual_income_manwon * STOCK_TOTAL_INCOME_MULT) * SAVE_RATE

            scenario_save = {
                scenarios[0]: save_1,
                scenarios[1]: save_2,
                scenarios[2]: save_3,
                scenarios[3]: save_4,
            }

            for s in scenarios:
                denom = max(float(scenario_save[s]), 1e-9)
                years_needed = int(np.ceil(float(house_price_manwon) / denom))
                out[reg][s][y] = max(1, years_needed)

    return out

data = build_data_real()

# ✅ 시뮬레이션 프레임(중간연도)에서 사용할 "해당 시점 기준연도" 매핑
def to_base_year(curr_year: int) -> int:
    if curr_year <= 2010:
        return 2010
    if curr_year <= 2015:
        return 2015
    if curr_year <= 2020:
        return 2020
    return 2025

earliest_purchase_year = {
    s: min(by + data[region][s][by] for by in base_years)
    for s in scenarios
}

PERSON_EMOJI = {
    scenarios[0]: "🏃‍♂️",
    scenarios[1]: "🦘",
    scenarios[2]: "💑",
    scenarios[3]: "📈",
}
HOUSE_EMOJI = "🏠"

LANE_DESC = {
    scenarios[0]: "1. 생활비를 제외한 금액으로 돈을 모은다면?, 저축률 50%",
    scenarios[1]: "2. 캥거루처럼 부모님께 의존하고 월급 전부는 집 사는 곳에 넣는다면?",
    scenarios[2]: "3. 맞벌이로 배우자와 함께 돈을 모은다면?",
    scenarios[3]: "4. 주식으로 ‘총소득=내소득×3’이 된다면?",
}

steps_between = 14
pause_frames = 3
jump_pause_frames = 1
shake_offsets = [0.35, -0.2, 0.0]

X_MIN, X_MAX = 2010, 2075

Q_X = 0.92
Q_DY = +0.5

def box_text(year: int, years_needed: int) -> str:
    highlight = "#FFD54A"
    return (
        f"<span style='font-size:14px;'>{year}년 기준</span>"
        f"<br>"
        f"<span style='font-size:20px; font-weight:800; color:{highlight};'>{years_needed}년</span>"
        f"<span style='font-size:14px;'> 걸림</span>"
    )

def scenario_boxes(label_year: int):
    anns = []
    for s in scenarios:
        y = lane_y[s]
        yrs = data[region][s][label_year]
        anns.append(
            dict(
                x=0.02, y=y, xref="paper", yref="y",
                text=box_text(label_year, yrs),
                showarrow=False,
                xanchor="left",
                yanchor="middle",
                align="left",
                font=dict(size=14, color="rgba(255,255,255,0.95)"),
                bgcolor="rgba(20,20,24,0.90)",
                bordercolor="rgba(255,255,255,0.18)",
                borderwidth=1,
                borderpad=10,
            )
        )
    return anns

def question_above_emoji():
    anns = []
    for s in scenarios:
        y = lane_y[s]
        anns.append(
            dict(
                x=Q_X, y=y + Q_DY,
                xref="paper", yref="y",        # ✅ x축이 아니라 화면 기준!
                text=f"<span style='font-size:13px; opacity:0.92;'><b>{LANE_DESC[s]}</b></span>",
                showarrow=False,
                xanchor="right",               # ✅ 오른쪽 기준으로 잡고
                yanchor="middle",
                align="right",                 # ✅ 글이 왼쪽으로 뻗게 해서 안 잘리게
                font=dict(size=30, color="rgba(255,255,255,0.90)"),
                bgcolor="rgba(0,0,0,0)",
)

        )
    return anns

def house_year_labels(label_year: int, house_x_map: dict):
    anns = []
    for s in scenarios:
        y = lane_y[s]
        yrs = data[region][s][label_year]
        hx = house_x_map[s]
        anns.append(
            dict(
                x=hx, y=y - 0.25, xref="x", yref="y",
                text=f"<b>{yrs}년</b>",
                showarrow=False,
                xanchor="center", yanchor="middle",
                font=dict(size=14, color="rgba(0,0,0,0.92)"),
                bgcolor="rgba(255,255,255,0.92)",
                bordercolor="rgba(0,0,0,0.25)",
                borderwidth=1,
                borderpad=5,
            )
        )
    return anns

# ✅ 시나리오별 최초 집 장만 연도(고정값) 사용
PURCHASE_YEAR = {s: min(by + data[region][s][by] for by in base_years) for s in scenarios}

def buy_labels(person_x: float):
    anns = []
    for s in scenarios:
        y = lane_y[s]
        done_year = float(PURCHASE_YEAR[s])
        if person_x >= done_year:  # ✅ 최초 장만 연도에 도달하면 뜸
            anns.append(
                dict(
                    x=float(person_x), y=y + 0.20,
                    xref="x", yref="y",
                    text="🏠 <b>집 장만!</b>",
                    showarrow=False,
                    xanchor="center", yanchor="middle",
                    font=dict(size=14, color="rgba(255,255,255,0.95)"),
                    bgcolor="rgba(0,0,0,0.82)",
                    bordercolor="rgba(255,255,255,0.22)",
                    borderwidth=1,
                    borderpad=7,
                )
            )
    return anns

def frame_annotations(person_x, house_x_map, label_year):
    return (
        scenario_boxes(label_year)
        + question_above_emoji()
        + house_year_labels(label_year, house_x_map)
        + buy_labels(person_x)
    )

init_year = 2010
init_person_x = 2010.0
init_house_map = {s: init_year + data[region][s][init_year] for s in scenarios}

traces = []

for s in scenarios:
    y = lane_y[s]
    traces.append(
        go.Scatter(
            x=[X_MIN, init_person_x], y=[y, y],
            mode="lines",
            line=dict(width=7, color="rgba(255,255,255,0.28)"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

for s in scenarios:
    y = lane_y[s]
    traces.append(
        go.Scatter(
            x=[init_person_x], y=[y],
            mode="text",
            text=[PERSON_EMOJI[s]],
            textfont=dict(size=34),
            showlegend=False,
            hoverinfo="skip",
        )
    )

for s in scenarios:
    y = lane_y[s]
    traces.append(
        go.Scatter(
            x=[init_house_map[s]], y=[y],
            mode="text",
            text=[HOUSE_EMOJI],
            textfont=dict(size=34),
            showlegend=False,
            hoverinfo="skip",
        )
    )

frames = []

def add_frame(person_x: float, house_x_map: dict, label_year: int):
    updates = []
    for _ in scenarios:
        updates.append(dict(x=[X_MIN, person_x]))
    for _ in scenarios:
        updates.append(dict(x=[person_x]))
    for s in scenarios:
        updates.append(dict(x=[house_x_map[s]]))

    frames.append(
        go.Frame(
            data=updates,
            traces=list(range(len(traces))),
            layout=go.Layout(annotations=frame_annotations(person_x, house_x_map, label_year)),
        )
    )

for i in range(len(base_years) - 1):
    start_year = base_years[i]
    end_year = base_years[i + 1]

    house_prev = {s: start_year + data[region][s][start_year] for s in scenarios}
    house_new  = {s: end_year   + data[region][s][end_year]   for s in scenarios}

    for t in np.linspace(0, 1, steps_between):
        person_x = start_year + (end_year - start_year) * t
        add_frame(person_x, house_prev, label_year=start_year)

    for _ in range(jump_pause_frames):
        add_frame(end_year, house_new, label_year=end_year)

    for off in shake_offsets:
        house_shake = {s: house_new[s] + off for s in scenarios}
        add_frame(end_year, house_shake, label_year=end_year)

    for _ in range(pause_frames):
        add_frame(end_year, house_new, label_year=end_year)

final_year = base_years[-1]
final_house = {s: final_year + data[region][s][final_year] for s in scenarios}
for off in shake_offsets:
    add_frame(final_year, {s: final_house[s] + off for s in scenarios}, label_year=final_year)
for _ in range(pause_frames):
    add_frame(final_year, final_house, label_year=final_year)

fig = go.Figure(data=traces, frames=frames)

fig.update_layout(
    height=820,
    paper_bgcolor="#0b0b0f",
    plot_bgcolor="#0b0b0f",
    font=dict(color="rgba(255,255,255,0.92)"),
    annotations=frame_annotations(init_person_x, init_house_map, init_year),

    xaxis=dict(
        autorange=False,          # ✅ 자동 OFF (축을 고정)
        range=[X_MAX, X_MIN],     # ✅ 2075 → 2010 (뒤집힌 축)
        tickmode="linear",
        tick0=X_MAX,              # ✅ 2075부터 눈금 생성
        dtick=1,
        title=dict(text="시간 흐름 (연도)", font=dict(size=16, color="rgba(255,255,255,0.88)")),
        showgrid=True,
        gridcolor="rgba(255,255,255,0.07)",
        zeroline=False,
        linecolor="rgba(255,255,255,0.15)",
        tickfont=dict(color="rgba(255,255,255,0.78)"),
        ),


    yaxis=dict(
        range=[-1.10, 4.10],
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        ticks="",
    ),

    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            bgcolor="rgba(20,20,24,0.92)",
            bordercolor="rgba(255,255,255,0.18)",
            borderwidth=1,
            buttons=[
                dict(
                    label="▶ 재생",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 80, "redraw": True},
                        "transition": {"duration": 0},
                        "fromcurrent": True
                    }]
                ),
                dict(
                    label="⏸ 정지",
                    method="animate",
                    args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                ),
            ],
            x=0.0, y=1.15, xanchor="left", yanchor="top"
        )
    ],

    margin=dict(l=35, r=30, t=115, b=55),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"지역: {region} | value=월소득(원) | 저축률(1)=50% | 캥거루(2)=100% | 맞벌이(3)=2배 | "
    f"주식(4)=총소득 3배 | 최초 집 장만 연도: { {s: int(earliest_purchase_year[s]) for s in scenarios} }"
)


# =========================================================
# 🏁 엔딩 섹션
# =========================================================
st.markdown("---")
st.header("🏁 결론: 데이터가 말해주는 것")

ending_image_path = BASE_DIR / "image2.png"

if ending_image_path.exists():
    st.image(
        Image.open(ending_image_path),
        caption="작은 선택의 차이가 만드는 미래",
        use_container_width=True
    )

st.markdown("### 📢 우리가 마주한 현실, 그리고 돌파구")

st.info("""
**“집 사기 힘든 세상입니다.”**  
하지만 데이터는 **절망이 아니라 전략의 근거**입니다.

- 소득만으로는 어렵다 → **성장의 속도**가 중요  
- 저축만으로는 부족하다 → **자본의 시간**이 필요  
- 혼자서는 길다 → **함께라면 현실이 된다**
""")

col_end1, col_end2, col_end3 = st.columns(3)

with col_end1:
    st.markdown("#### 🚀 Self-Growth")
    st.caption("소득 성장")
    st.write("연봉 상승률 2~3%의 차이가 10년 후 자산 격차를 만듭니다.")

with col_end2:
    st.markdown("#### 💰 Investment")
    st.caption("자본 활용")
    st.write("단순 저축을 넘어 자산이 일하게 해야 합니다.")

with col_end3:
    st.markdown("#### 🤝 Partnership")
    st.caption("함께의 힘")
    st.write("맞벌이는 내 집 마련 기간을 구조적으로 단축시킵니다.")

st.subheader("🌟 당신의 시나리오는 지금부터입니다")

if st.button("🚀 내 집 마련 시나리오 시작하기"):
    st.balloons()
    st.success("데이터를 이해한 순간, 선택은 이미 달라졌습니다.")

st.divider()

# =========================
# 12) (4) 테이블
# =========================
st.subheader(" 🧾 집계 데이터 테이블")

show_raw = st.checkbox("원자료(소득/아파트 대표값)도 같이 보기", value=True)

if show_raw:
    out = df[[
        "Year",
        "income_value",
        "income_monthly_won",
        "apt_price_median",
        "Income_Index",
        "Apartment_Index",
        "Purchasable_Pyeong",
        "Months_for_1Pyeong",
        "apt_price_col_used"
    ]].copy()

    out = out.rename(columns={
        "income_value": "소득(value)",
        "income_monthly_won": "월소득(환산)",
        "apt_price_median": "아파트(전국대표)_평당가격",
        "Income_Index": "소득지수(2010=100)",
        "Apartment_Index": "아파트지수(2010=100)",
        "Purchasable_Pyeong": "구매가능평수(월소득/평당)",
        "Months_for_1Pyeong": "1평구매_개월수(평당/월소득)",
        "apt_price_col_used": "아파트가격_컬럼"
    })
else:
    out = df[["Year", "Income_Index", "Apartment_Index", "Purchasable_Pyeong", "Months_for_1Pyeong"]].copy()

st.dataframe(out.set_index("Year"), use_container_width=True)


