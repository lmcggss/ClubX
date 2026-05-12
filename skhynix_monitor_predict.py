import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.linear_model import LinearRegression


@dataclass
class PredictionResult:
    next_close: float
    trend: str
    confidence_note: str


def download_prices(ticker: str = "000660.KS", period: str = "2y") -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError("주가 데이터를 가져오지 못했습니다. 티커 또는 네트워크 상태를 확인하세요.")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    feat["ret_1d"] = feat["Close"].pct_change()
    feat["ma_5"] = feat["Close"].rolling(5).mean()
    feat["ma_20"] = feat["Close"].rolling(20).mean()
    feat["vol_5"] = feat["ret_1d"].rolling(5).std()
    feat["mom_5"] = feat["Close"] / feat["Close"].shift(5) - 1

    # 내일 종가를 예측하기 위해 target을 하루 앞으로 이동
    feat["target_next_close"] = feat["Close"].shift(-1)
    return feat.dropna()


def train_and_predict(featured_df: pd.DataFrame) -> PredictionResult:
    cols = ["Close", "ret_1d", "ma_5", "ma_20", "vol_5", "mom_5", "Volume"]
    X = featured_df[cols]
    y = featured_df["target_next_close"]

    model = LinearRegression()
    model.fit(X, y)

    last_row = X.iloc[[-1]]
    pred = float(model.predict(last_row)[0])
    last_close = float(featured_df["Close"].iloc[-1])

    change_pct = (pred / last_close - 1) * 100
    if change_pct > 0.5:
        trend = "상승 가능성"
    elif change_pct < -0.5:
        trend = "하락 가능성"
    else:
        trend = "보합 가능성"

    confidence_note = (
        "단순 선형회귀 기반 참고치입니다. 실거래 의사결정에는 "
        "재무지표/뉴스/거시환경을 반드시 함께 반영하세요."
    )

    return PredictionResult(next_close=pred, trend=trend, confidence_note=confidence_note)


def monitor_loop(ticker: str, interval_sec: int = 300) -> None:
    print(f"[{ticker}] 실시간 모니터링 시작 (Ctrl+C 종료)")
    while True:
        try:
            df = download_prices(ticker=ticker, period="6mo")
            featured = make_features(df)
            result = train_and_predict(featured)

            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
            last_close = featured["Close"].iloc[-1]
            print("-" * 70)
            print(f"시각: {now}")
            print(f"최근 종가: {last_close:,.0f} KRW")
            print(f"내일 예측 종가: {result.next_close:,.0f} KRW")
            print(f"예측 흐름: {result.trend}")
            print(f"주의: {result.confidence_note}")
        except Exception as e:
            print(f"오류: {e}")

        import time

        time.sleep(interval_sec)


def run_once(ticker: str) -> None:
    df = download_prices(ticker=ticker, period="2y")
    featured = make_features(df)
    result = train_and_predict(featured)

    last_date = featured.index[-1].date()
    next_date = (pd.Timestamp(last_date) + timedelta(days=1)).date()
    last_close = featured["Close"].iloc[-1]

    print(f"종목: {ticker}")
    print(f"기준일: {last_date}")
    print(f"예측대상일(다음 거래일 추정): {next_date}")
    print(f"최근 종가: {last_close:,.0f} KRW")
    print(f"내일 예측 종가: {result.next_close:,.0f} KRW")
    print(f"예측 흐름: {result.trend}")
    print(f"주의: {result.confidence_note}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SK하이닉스 주가 모니터링 + 다음 거래일 흐름 예측")
    parser.add_argument("--ticker", default="000660.KS", help="예: 000660.KS (SK하이닉스)")
    parser.add_argument("--monitor", action="store_true", help="주기적으로 모니터링")
    parser.add_argument("--interval", type=int, default=300, help="모니터링 주기(초)")

    args = parser.parse_args()

    if args.monitor:
        monitor_loop(args.ticker, args.interval)
    else:
        run_once(args.ticker)
