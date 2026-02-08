#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import random
import re
import time
from typing import Dict, Any, Callable, TypeVar, Optional

import yfinance as yf
import pandas as pd
from yfinance import exceptions as yf_exceptions


T = TypeVar("T")


def _retry_call(fn: Callable[[], T], max_retries: int, base_wait: float) -> T:
    last_err = None
    for i in range(max_retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            is_rate = isinstance(e, yf_exceptions.YFRateLimitError)
            if i >= max_retries:
                break
            # Exponential backoff with jitter
            wait = base_wait * (2 ** i) + random.uniform(0, base_wait)
            if is_rate:
                time.sleep(wait)
            else:
                time.sleep(min(wait, 10.0))
    raise last_err


def fetch_history(symbol: str, years: int, max_retries: int, base_wait: float) -> pd.DataFrame:
    period = f"{years}y"
    def _call():
        return yf.Ticker(symbol).history(period=period, interval="1d", auto_adjust=False)

    df = _retry_call(_call, max_retries=max_retries, base_wait=base_wait)
    if df.empty:
        raise RuntimeError(f"No price history returned for {symbol} with period={period}")
    df = df.dropna(subset=["Close"]).copy()
    return df


def compute_kline_summary(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA250"] = df["Close"].rolling(250).mean()
    df["VOL20"] = df["Volume"].rolling(20).mean()
    df["VOL60"] = df["Volume"].rolling(60).mean()

    last = df.iloc[-1]
    last_date = df.index[-1].date()

    # 52-week range (approx 252 trading days)
    tail = df.tail(252)
    high_52w = float(tail["High"].max())
    low_52w = float(tail["Low"].min())
    last_close = float(last["Close"])
    pct_from_high = (last_close / high_52w - 1.0) * 100.0
    pct_from_low = (last_close / low_52w - 1.0) * 100.0

    ma20 = float(last["MA20"]) if pd.notna(last["MA20"]) else None
    ma60 = float(last["MA60"]) if pd.notna(last["MA60"]) else None
    ma250 = float(last["MA250"]) if pd.notna(last["MA250"]) else None

    trend = "震荡"
    if ma60 and ma250:
        if last_close > ma60 > ma250:
            trend = "上升"
        elif last_close < ma60 < ma250:
            trend = "下降"

    vol20 = float(last["VOL20"]) if pd.notna(last["VOL20"]) else None
    vol60 = float(last["VOL60"]) if pd.notna(last["VOL60"]) else None
    vol_ratio = (vol20 / vol60) if (vol20 and vol60) else None

    return {
        "last_date": last_date,
        "last_close": last_close,
        "ma20": ma20,
        "ma60": ma60,
        "ma250": ma250,
        "trend": trend,
        "high_52w": high_52w,
        "low_52w": low_52w,
        "pct_from_high": pct_from_high,
        "pct_from_low": pct_from_low,
        "vol_ratio": vol_ratio,
    }


def fetch_snapshot(symbol: str, max_retries: int, base_wait: float) -> Dict[str, Any]:
    t = yf.Ticker(symbol)
    def _call():
        return t.info or {}

    info = _retry_call(_call, max_retries=max_retries, base_wait=base_wait)

    snapshot = {
        "price": info.get("regularMarketPrice"),
        "currency": info.get("currency"),
        "market_cap": info.get("marketCap"),
        "shares_out": info.get("sharesOutstanding"),
        "trailing_pe": info.get("trailingPE"),
        "price_to_book": info.get("priceToBook"),
        "dividend_yield": info.get("dividendYield"),
        "as_of": info.get("regularMarketTime"),
    }
    return snapshot


def fmt_money(num: Any, unit: str = "") -> str:
    if num is None:
        return "N/A"
    try:
        return f"{num:,.2f}{unit}"
    except Exception:
        return str(num)


def fmt_pct(num: Any) -> str:
    if num is None:
        return "N/A"
    return f"{num:.2f}%"

def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None

def compute_valuation_curves(symbol: str, price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build rough historical PE/PB/PS using yfinance financials.
    For non-US tickers, fields may be missing; return empty if not enough data.
    """
    t = yf.Ticker(symbol)
    info = t.info or {}
    shares = _safe_float(info.get("sharesOutstanding"))
    if not shares or shares <= 0:
        return pd.DataFrame()

    # Quarterly financials (last 4 quarters) for TTM approximations
    q_fin = t.quarterly_financials
    q_bs = t.quarterly_balance_sheet

    # Need at least revenue and net income for PS/PE, and total equity for PB
    if q_fin is None or q_fin.empty:
        return pd.DataFrame()

    # yfinance uses columns as period end dates
    q_fin = q_fin.copy()
    q_bs = q_bs.copy() if q_bs is not None else pd.DataFrame()

    # Map common field names
    def pick_row(df: pd.DataFrame, candidates: list) -> Optional[pd.Series]:
        for name in candidates:
            if name in df.index:
                return df.loc[name]
        return None

    net_income = pick_row(q_fin, ["Net Income", "Net Income Applicable To Common Shares"])
    total_revenue = pick_row(q_fin, ["Total Revenue", "TotalRevenue"])
    total_equity = pick_row(q_bs, ["Total Stockholder Equity", "Total Stockholder Equity (Total)"])

    if net_income is None and total_revenue is None and total_equity is None:
        return pd.DataFrame()

    # Build TTM approximations using sum of latest 4 quarters
    net_income_ttm = net_income.sum() if net_income is not None else None
    revenue_ttm = total_revenue.sum() if total_revenue is not None else None
    equity_latest = total_equity.iloc[0] if total_equity is not None else None

    if net_income_ttm is None and revenue_ttm is None and equity_latest is None:
        return pd.DataFrame()

    price = price_df[["Close"]].copy()
    price["PE"] = None
    price["PB"] = None
    price["PS"] = None

    if net_income_ttm and net_income_ttm != 0:
        eps_ttm = net_income_ttm / shares
        price["PE"] = price["Close"] / eps_ttm
    if equity_latest and equity_latest != 0:
        bps = equity_latest / shares
        price["PB"] = price["Close"] / bps
    if revenue_ttm and revenue_ttm != 0:
        sps = revenue_ttm / shares
        price["PS"] = price["Close"] / sps

    return price[["PE", "PB", "PS"]]


def save_valuation_svg(df: pd.DataFrame, out_path: str, title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    if df is None or df.empty:
        return False

    plt.figure(figsize=(10, 4))
    for col, color in [("PE", "#1f77b4"), ("PB", "#ff7f0e"), ("PS", "#2ca02c")]:
        if col in df.columns and df[col].notna().any():
            plt.plot(df.index, df[col], label=col, linewidth=1.2, color=color)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, format="svg")
    plt.close()
    return True


def format_snapshot_md(symbol: str, snap: Dict[str, Any]) -> str:
    as_of = "N/A"
    if snap.get("as_of"):
        as_of = dt.datetime.fromtimestamp(int(snap["as_of"])).date().isoformat()

    dividend_yield = snap.get("dividend_yield")
    if dividend_yield is not None:
        dividend_yield = dividend_yield * 100.0

    return (
        f"- 行情快照（{as_of}）：股价约 {fmt_money(snap.get('price'))} {snap.get('currency') or ''}。\n"
        f"- 总市值：{fmt_money(snap.get('market_cap'))}。\n"
        f"- 总股本：{fmt_money(snap.get('shares_out'))}。\n"
        f"- 市盈率（TTM）：{fmt_money(snap.get('trailing_pe'))}。\n"
        f"- 市净率：{fmt_money(snap.get('price_to_book'))}。\n"
        f"- 股息率：{fmt_pct(dividend_yield)}。\n"
    )


def format_kline_md(k: Dict[str, Any]) -> str:
    ma_line = (
        f"- 均线：MA20={fmt_money(k['ma20'])}，MA60={fmt_money(k['ma60'])}，"
        f"MA250={fmt_money(k['ma250'])}。"
    )
    vol_line = "- 量能：近20日/60日均量比={}.".format(
        f"{k['vol_ratio']:.2f}" if k["vol_ratio"] is not None else "N/A"
    )

    return (
        f"- 数据截至：{k['last_date']}。\n"
        f"- 收盘价：{fmt_money(k['last_close'])}。\n"
        f"- 中期趋势：{k['trend']}。\n"
        f"{ma_line}\n"
        f"- 52周区间：高点 {fmt_money(k['high_52w'])}，低点 {fmt_money(k['low_52w'])}。\n"
        f"- 距52周高点：{fmt_pct(k['pct_from_high'])}；距52周低点：{fmt_pct(k['pct_from_low'])}。\n"
        f"{vol_line}\n"
    )


def update_report(report_path: str, snapshot_md: str, kline_md: str) -> None:
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    def replace_section(content: str, header: str, new_body: str) -> str:
        pattern = rf"(## {re.escape(header)}\n)([\s\S]*?)(?=\n## |\Z)"
        m = re.search(pattern, content)
        if not m:
            raise RuntimeError(f"Section not found: {header}")
        return content[: m.start(2)] + new_body + content[m.end(2) :]

    content = replace_section(
        content,
        "15. 公司当前市值",
        "".join([
            "- 行情快照：\n",
            snapshot_md,
            "\n",
        ]),
    )
    content = replace_section(
        content,
        "16. 各类估值曲线",
        "".join([
            "- 当前估值快照：\n",
            snapshot_md,
            "- 估值曲线（5年PE/PB/PS）：见生成的 SVG。\n\n",
        ]),
    )
    content = replace_section(
        content,
        "17. K 线基本分析",
        "".join([
            kline_md,
            "\n",
        ]),
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="600519.SS")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--outdir", default="/Users/qqqzzz/tmp/finance/output")
    parser.add_argument("--report", default="")
    parser.add_argument("--history-csv", default="", help="Use local CSV instead of downloading")
    parser.add_argument("--no-snapshot", action="store_true", help="Skip live snapshot fetch")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-wait", type=float, default=2.0)
    parser.add_argument("--svg", action="store_true", help="Generate valuation SVG (PE/PB/PS)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.history_csv:
        df = pd.read_csv(args.history_csv, index_col=0, parse_dates=True)
        if df.empty or "Close" not in df.columns:
            raise RuntimeError("Local CSV is empty or missing required columns")
    else:
        df = fetch_history(args.symbol, args.years, args.max_retries, args.retry_wait)
    csv_path = os.path.join(args.outdir, f"{args.symbol}_price_{args.years}y.csv")
    df.to_csv(csv_path)

    kline = compute_kline_summary(df)
    snapshot = {} if args.no_snapshot else fetch_snapshot(args.symbol, args.max_retries, args.retry_wait)

    snippet = (
        "## 行情与K线摘要\n"
        f"{format_snapshot_md(args.symbol, snapshot)}\n"
        "## K线基本分析\n"
        f"{format_kline_md(kline)}\n"
    )

    snippet_path = os.path.join(args.outdir, f"{args.symbol}_kline_summary.md")
    with open(snippet_path, "w", encoding="utf-8") as f:
        f.write(snippet)

    if args.svg:
        val_df = compute_valuation_curves(args.symbol, df)
        svg_path = os.path.join(args.outdir, f"{args.symbol}_valuation_{args.years}y.svg")
        saved = save_valuation_svg(val_df, svg_path, f"{args.symbol} 估值曲线 (PE/PB/PS)")
        if saved:
            print(f"Saved valuation SVG: {svg_path}")
        else:
            print("Valuation SVG not generated (missing data or matplotlib).")

    if args.report:
        update_report(args.report, format_snapshot_md(args.symbol, snapshot), format_kline_md(kline))

    print(f"Saved price history: {csv_path}")
    print(f"Saved summary snippet: {snippet_path}")
    if args.report:
        print(f"Updated report: {args.report}")


if __name__ == "__main__":
    main()
