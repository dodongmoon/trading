"""
통합 백테스팅 UI - 모든 지표를 조합한 복합 전략 백테스팅

이 모듈은 RSI, MACD, 볼린저밴드, 이동평균선을 자유롭게 조합하여
복합 매매 전략을 백테스팅할 수 있는 웹 인터페이스를 제공합니다.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import os
import requests
import json

# 우리가 만든 모듈들 import
from extended_data_collector import ExtendedBitgetDataCollector
from indicators_rsi import RSIIndicator
from indicators_macd import MACDIndicator
from indicators_bollinger import BollingerBandsIndicator
from indicators_moving_average import MovingAverageIndicator


class IntegratedBacktestingEngine:
    """통합 백테스팅 엔진"""
    
    def __init__(self):
        self.data_collector = ExtendedBitgetDataCollector()
    
    def calculate_all_indicators(self, df, indicator_configs):
        """
        모든 선택된 지표 계산
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            indicator_configs (dict): 지표 설정
            
        Returns:
            pd.DataFrame: 모든 지표가 계산된 데이터프레임
        """
        df_result = df.copy()
        
        # RSI 계산
        if indicator_configs.get('use_rsi', False):
            rsi = RSIIndicator(
                period=indicator_configs.get('rsi_period', 14)
            )
            df_result = rsi.calculate_rsi(df_result)
            df_result = rsi.analyze_rsi_signals(
                df_result,
                oversold=indicator_configs.get('rsi_oversold', 30),
                overbought=indicator_configs.get('rsi_overbought', 70)
            )
        
        # MACD 계산
        if indicator_configs.get('use_macd', False):
            macd = MACDIndicator(
                fast_period=indicator_configs.get('macd_fast', 12),
                slow_period=indicator_configs.get('macd_slow', 26),
                signal_period=indicator_configs.get('macd_signal', 9)
            )
            df_result = macd.calculate_macd(df_result)
            df_result = macd.analyze_macd_signals(df_result)
        
        # 볼린저 밴드 계산
        if indicator_configs.get('use_bollinger', False):
            bb = BollingerBandsIndicator(
                period=indicator_configs.get('bb_period', 20),
                std_dev=indicator_configs.get('bb_std', 2)
            )
            df_result = bb.calculate_bollinger_bands(df_result)
            df_result = bb.analyze_bollinger_signals(df_result)
        
        # 이동평균선 계산
        if indicator_configs.get('use_ma', False):
            ma = MovingAverageIndicator(
                short_period=indicator_configs.get('ma_short', 20),
                long_period=indicator_configs.get('ma_long', 50),
                ma_type=indicator_configs.get('ma_type', 'sma')
            )
            df_result = ma.calculate_moving_averages(df_result)
            df_result = ma.analyze_ma_signals(df_result)
        
        return df_result
    
    def generate_combined_signals(self, df, strategy_config):
        """
        복합 전략 신호 생성
        
        Args:
            df (pd.DataFrame): 모든 지표가 계산된 데이터프레임
            strategy_config (dict): 전략 설정
            
        Returns:
            pd.DataFrame: 복합 신호가 추가된 데이터프레임
        """
        df_result = df.copy()
        
        # 복합 신호 초기화
        df_result['Combined_Buy_Signal'] = False
        df_result['Combined_Sell_Signal'] = False
        df_result['Combined_Signal_Type'] = 'hold'
        df_result['Signal_Strength'] = 0  # 신호 강도 (0-4)
        
        # 각 지표별 신호 수집
        buy_conditions = []
        sell_conditions = []
        
        # RSI 신호
        if strategy_config.get('use_rsi', False):
            if 'RSI_Buy_Signal' in df_result.columns:
                buy_conditions.append(df_result['RSI_Buy_Signal'])
            if 'RSI_Sell_Signal' in df_result.columns:
                sell_conditions.append(df_result['RSI_Sell_Signal'])
        
        # MACD 신호
        if strategy_config.get('use_macd', False):
            if 'MACD_Buy_Signal' in df_result.columns:
                buy_conditions.append(df_result['MACD_Buy_Signal'])
            if 'MACD_Sell_Signal' in df_result.columns:
                sell_conditions.append(df_result['MACD_Sell_Signal'])
        
        # 볼린저 밴드 신호
        if strategy_config.get('use_bollinger', False):
            if 'BB_Buy_Signal' in df_result.columns:
                buy_conditions.append(df_result['BB_Buy_Signal'])
            if 'BB_Sell_Signal' in df_result.columns:
                sell_conditions.append(df_result['BB_Sell_Signal'])
        
        # 이동평균선 신호
        if strategy_config.get('use_ma', False):
            if 'MA_Buy_Signal' in df_result.columns:
                buy_conditions.append(df_result['MA_Buy_Signal'])
            if 'MA_Sell_Signal' in df_result.columns:
                sell_conditions.append(df_result['MA_Sell_Signal'])
        
        # 전략 로직 적용
        strategy_logic = strategy_config.get('logic', 'any')  # 'any' 또는 'all'
        min_signals = strategy_config.get('min_signals', 1)
        
        if buy_conditions:
            if strategy_logic == 'all':
                # 모든 선택된 지표가 매수 신호
                combined_buy = pd.concat(buy_conditions, axis=1).all(axis=1)
            else:  # 'any' 또는 최소 신호 개수
                # 최소 N개 이상의 지표가 매수 신호
                signal_count = pd.concat(buy_conditions, axis=1).sum(axis=1)
                combined_buy = signal_count >= min_signals
                df_result['Signal_Strength'] = signal_count
        else:
            combined_buy = pd.Series(False, index=df_result.index)
        
        if sell_conditions:
            if strategy_logic == 'all':
                # 모든 선택된 지표가 매도 신호
                combined_sell = pd.concat(sell_conditions, axis=1).all(axis=1)
            else:  # 'any' 또는 최소 신호 개수
                # 최소 N개 이상의 지표가 매도 신호
                signal_count = pd.concat(sell_conditions, axis=1).sum(axis=1)
                combined_sell = signal_count >= min_signals
                if 'Signal_Strength' not in df_result.columns:
                    df_result['Signal_Strength'] = signal_count
        else:
            combined_sell = pd.Series(False, index=df_result.index)
        
        # 복합 신호 적용
        df_result.loc[combined_buy, 'Combined_Buy_Signal'] = True
        df_result.loc[combined_buy, 'Combined_Signal_Type'] = 'buy'
        
        df_result.loc[combined_sell, 'Combined_Sell_Signal'] = True
        df_result.loc[combined_sell, 'Combined_Signal_Type'] = 'sell'
        
        return df_result
    
    def run_backtest(self, df, initial_capital=10000, trade_ratio=1.0, leverage=1.0):
        """
        백테스팅 실행
        
        Args:
            df (pd.DataFrame): 신호가 생성된 데이터프레임
            initial_capital (float): 초기 자본
            trade_ratio (float): 거래 비율 (0.0-1.0)
            
        Returns:
            tuple: (백테스팅 결과 DataFrame, 성과 지표 dict)
        """
        df_backtest = df.copy()
        
        # 백테스팅 변수 초기화
        portfolio_value = initial_capital
        cash = initial_capital
        position = 0.0  # 보유 수량 (계약 수/코인 수)
        invested_amount = 0.0  # 누적 증거금(마진) 합계
        avg_entry_price = 0.0   # 평균 진입가 (레버리지 포지션 기준)
        allocation_unit = initial_capital * trade_ratio  # 1회 매수 목표 금액
        trades = []
        portfolio_values = [initial_capital]
        liquidated = False
        liquidation_info = {}
        
        # 거래 실행
        for i in range(1, len(df_backtest)):
            current_price = df_backtest['close'].iloc[i]
            current_low = df_backtest['low'].iloc[i]
            current_date = df_backtest.index[i]

            # 레버리지 포지션 청산 체크 (롱 전용)
            if leverage > 1.0 and position > 0.0 and avg_entry_price > 0.0:
                liquidation_price = avg_entry_price * (1.0 - (1.0 / float(leverage)))
                if current_low <= liquidation_price:
                    # 청산: 포트폴리오 전액 손실 처리(요청 사양)
                    liquidated = True
                    liquidation_info = {
                        'date': current_date,
                        'price': float(liquidation_price),
                        'leverage': float(leverage)
                    }
                    portfolio_value = 0.0
                    cash = 0.0
                    position = 0.0
                    invested_amount = 0.0
                    avg_entry_price = 0.0
                    portfolio_values.append(portfolio_value)
                    # 남은 구간 0으로 채우기
                    remaining = (len(df_backtest) - 1) - i
                    if remaining > 0:
                        portfolio_values.extend([0.0] * remaining)
                    break
            
            # 매수 신호 (부분 매수 허용, 남은 현금이 목표보다 작으면 전액 매수)
            if df_backtest['Combined_Buy_Signal'].iloc[i] and cash > 0:
                desired_amount = allocation_unit
                trade_margin = min(cash, desired_amount)  # 증거금(마진)
                if trade_margin > 0:
                    # 레버리지 적용: 계약 수 = (마진 * 레버리지) / 가격
                    qty = (trade_margin * float(leverage)) / current_price
                    # 평균 진입가 갱신(수량 가중)
                    if position > 0.0:
                        new_total_qty = position + qty
                        avg_entry_price = ((avg_entry_price * position) + (current_price * qty)) / new_total_qty
                        position = new_total_qty
                    else:
                        position = qty
                        avg_entry_price = current_price

                    cash -= trade_margin
                    invested_amount += trade_margin
                    trades.append({
                        'type': 'buy',
                        'date': current_date,
                        'price': float(current_price),
                        'quantity': float(qty),
                        'amount': float(trade_margin),  # 마진 규모 기록
                        'leverage': float(leverage),
                        'signal_strength': df_backtest['Signal_Strength'].iloc[i] if 'Signal_Strength' in df_backtest.columns else 1
                    })

            # 매도 신호 (보유 수량 있을 때만 전량 청산)
            if df_backtest['Combined_Sell_Signal'].iloc[i] and position > 0:
                # 선물 PnL = 수량 * (현재가 - 평균진입가)
                realized_pnl = position * (current_price - avg_entry_price)
                # 마진 반환 + 실현손익 반영
                cash += invested_amount + realized_pnl
                trades.append({
                    'type': 'sell',
                    'date': current_date,
                    'price': float(current_price),
                    'quantity': float(position),
                    'amount': float(invested_amount + realized_pnl),
                    'leverage': float(leverage),
                    'signal_strength': df_backtest['Signal_Strength'].iloc[i] if 'Signal_Strength' in df_backtest.columns else 1
                })
                position = 0.0
                invested_amount = 0.0
                avg_entry_price = 0.0
            
            # 포트폴리오 가치 계산
            # 현재 포지션의 미실현 손익
            unrealized_pnl = 0.0
            if position > 0.0 and avg_entry_price > 0.0:
                unrealized_pnl = position * (current_price - avg_entry_price)
            # 총자산 = 가용현금 + (보유 마진) + 미실현손익
            portfolio_value = cash + invested_amount + unrealized_pnl
            portfolio_values.append(portfolio_value)
        
        # 백테스팅 결과 DataFrame에 추가
        # 청산으로 루프 중단 시 길이 보정됨. 그 외에도 길이가 맞는지 보정
        if len(portfolio_values) != len(df_backtest):
            # 마지막 값으로 패딩
            last_val = portfolio_values[-1] if portfolio_values else initial_capital
            portfolio_values += [last_val] * (len(df_backtest) - len(portfolio_values))
        df_backtest['Portfolio_Value'] = portfolio_values
        
        # 성과 지표 계산
        if liquidated:
            performance_metrics = {
                'total_trades': len([t for t in trades if t['type'] == 'sell']),
                'total_return': -(initial_capital),
                'total_return_pct': -100.0,
                'win_rate': 0,
                'avg_profit': 0,
                'max_profit': 0,
                'max_loss': -initial_capital,
                'trades_per_month': 0,
                'avg_holding_period': 0,
                'final_portfolio_value': 0.0,
                'liquidated': True,
                'liquidation_date': liquidation_info.get('date'),
                'liquidation_price': liquidation_info.get('price')
            }
        elif len(trades) >= 2:
            performance_metrics = self.calculate_performance_metrics(
                trades, df_backtest, initial_capital
            )
        else:
            performance_metrics = {
                'total_trades': len(trades),
                'total_return': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'max_profit': 0,
                'max_loss': 0,
                'trades_per_month': 0,
                'avg_holding_period': 0,
                'final_portfolio_value': portfolio_value,
                'liquidated': False
            }
        
        return df_backtest, performance_metrics, trades
    
    def calculate_performance_metrics(self, trades, df, initial_capital):
        """성과 지표 계산"""
        if len(trades) < 2:
            return {}
        
        # 거래 쌍 생성 (매수-매도)
        trade_pairs = []
        for i in range(0, len(trades) - 1, 2):
            if i + 1 < len(trades) and trades[i]['type'] == 'buy' and trades[i + 1]['type'] == 'sell':
                buy_trade = trades[i]
                sell_trade = trades[i + 1]
                
                profit = sell_trade['amount'] - buy_trade['amount']
                profit_pct = (profit / buy_trade['amount']) * 100
                holding_period = (sell_trade['date'] - buy_trade['date']).days
                
                trade_pairs.append({
                    'profit': profit,
                    'profit_pct': profit_pct,
                    'holding_period': holding_period,
                    'buy_date': buy_trade['date'],
                    'sell_date': sell_trade['date'],
                    'buy_price': buy_trade['price'],
                    'sell_price': sell_trade['price']
                })
        
        if not trade_pairs:
            return {'total_trades': len(trades)}
        
        # 성과 지표 계산
        profits = [tp['profit'] for tp in trade_pairs]
        profit_pcts = [tp['profit_pct'] for tp in trade_pairs]
        holding_periods = [tp['holding_period'] for tp in trade_pairs]
        
        total_return = sum(profits)
        final_value = float(df['Portfolio_Value'].iloc[-1])
        total_return_pct = ((final_value / initial_capital) - 1) * 100
        
        win_trades = [p for p in profits if p > 0]
        win_rate = (len(win_trades) / len(profits)) * 100 if profits else 0
        
        avg_profit = np.mean(profits) if profits else 0
        max_profit = max(profits) if profits else 0
        max_loss = min(profits) if profits else 0
        
        # 거래 빈도 (월평균) 및 보유기간
        period_days = max((df.index[-1] - df.index[0]).total_seconds() / 86400.0, 1e-9)
        trades_per_month = (len(trade_pairs) / period_days) * 30.0 if period_days > 0 else 0.0
        
        avg_holding_period = np.mean(holding_periods) if holding_periods else 0
        
        return {
            'total_trades': len(trade_pairs),
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'trades_per_month': trades_per_month,
            'avg_holding_period': avg_holding_period,
            'final_portfolio_value': final_value
        }


def create_integrated_backtesting_ui():
    """통합 백테스팅 UI 생성"""
    
    st.set_page_config(
        page_title="🚀 통합 백테스팅 시스템",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # 호스팅 환경에서 메인 컨테이너 폭이 축소되는 현상 방지(TradingView 포함 전역 폭 보정)
    st.markdown(
        """
        <style>
        /* 메인 컨텐츠 최대 폭 확장 */
        div.block-container{max-width: 1600px !important;}
        /* 임베드 컨테이너는 항상 가로 100% 차지 */
        .tradingview-widget-container{width:100% !important; max-width:100% !important;}
        .tradingview-widget-container__widget{width:100% !important;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    st.title("🚀 통합 백테스팅 시스템")
    st.markdown("**RSI, MACD, 볼린저밴드, 이동평균선을 자유롭게 조합한 복합 전략 백테스팅**")
    
    # 사이드바 - 기본 설정
    st.sidebar.header("📊 기본 설정")
    
    symbol = st.sidebar.selectbox(
        "거래 심볼",
        options=["BTCUSDT", "ETHUSDT", "ADAUSDT", "SOLUSDT"],
        index=0
    )
    
    timeframe = st.sidebar.selectbox(
        "시간 간격",
        options=["1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d"],
        index=5
    )

    # 심볼 매핑 (요청: 백테스팅은 항상 선물, TV는 참고용 Spot 고정)
    base_asset = symbol.replace("USDT", "")
    ccxt_symbol = f"{base_asset}/USDT:USDT"  # Bitget USDT-M 선물 심볼
    tradingview_symbol = f"BITGET:{symbol}"  # TV는 Spot 고정 참고용
    
    # API 제한 정보 로드 (timeframe 변경 시 자동 반영)
    try:
        with open('api_limits_3x_config.json', 'r') as f:
            limits_payload = json.load(f)
        split_limits = limits_payload.get('split_3x_limits', {})
        tf_limits = split_limits.get(timeframe, {})
        max_candles_per_call = tf_limits.get('max_candles_per_call')
        max_calls = tf_limits.get('max_calls', 3)
        total_max_candles = tf_limits.get('total_max_candles')
        total_max_days = tf_limits.get('total_max_days')
    except Exception:
        split_limits = {}
        tf_limits = {}
        max_candles_per_call = None
        max_calls = 3
        total_max_candles = None
        total_max_days = None

    # 파일이 없거나 키가 없을 때의 안전 계산 (원시 한도 → 95% → 3회 호출)
    if max_candles_per_call is None or total_max_candles is None or total_max_days is None:
        # 원시 한도 (비트겟 실측 기준)
        raw_per_call_limit = {
            '1m': 1000,
            '3m': 1000,
            '5m': 1000,
            '15m': 1000,
            '30m': 1000,
            '1h': 1000,
            '4h': 540,
            '1d': 90,
        }.get(timeframe, 1000)
        safety = 0.95
        max_candles_per_call = int(raw_per_call_limit * safety)
        total_max_candles = max_candles_per_call * max_calls

        candles_per_day = {
            '1m': (24 * 60) // 1,
            '3m': (24 * 60) // 3,
            '5m': (24 * 60) // 5,
            '15m': (24 * 60) // 15,
            '30m': (24 * 60) // 30,
            '1h': (24 * 60) // 60,
            '4h': (24 * 60) // 240,
            '1d': 1,
        }.get(timeframe, 24)
        total_max_days = total_max_candles / candles_per_day

    max_candles = int(total_max_candles)

    # 기간 선택 UI: 초단기(1m/3m/5m)는 분/시간/일 단위 제공, 그 외는 일 단위 유지
    period_label = ""
    if timeframe in ["1m", "3m", "5m"]:
        tf_min = {"1m": 1, "3m": 3, "5m": 5}[timeframe]
        total_max_minutes = int(max_candles * tf_min)
        total_max_hours = max(1, total_max_minutes // 60)
        total_max_days_float = total_max_minutes / (60 * 24)

        units = ["분", "시간"] if timeframe == "1m" else ["분", "시간", "일"]
        unit = st.sidebar.selectbox("기간 단위", units, index=1)
        if unit == "분":
            # 최소 30분, 스텝은 타임프레임 간격
            minutes = st.sidebar.slider(
                f"백테스팅 기간 (분) - 최대 {total_max_minutes}분",
                min_value=max(tf_min, 30),
                max_value=max(tf_min, total_max_minutes),
                value=min(720, total_max_minutes),
                step=tf_min
            )
            days = minutes / (60 * 24)
            period_label = f"{minutes}분"
        elif unit == "시간":
            hours = st.sidebar.slider(
                f"백테스팅 기간 (시간) - 최대 {total_max_hours}시간",
                min_value=1,
                max_value=total_max_hours,
                value=min(24, total_max_hours),
                step=1
            )
            days = hours / 24.0
            period_label = f"{hours}시간"
        else:
            max_days_int = max(1, int(total_max_days_float))
            days_int = st.sidebar.slider(
                f"백테스팅 기간 (일) - 최대 {max_days_int}일",
                min_value=1,
                max_value=max_days_int,
                value=min(7, max_days_int),
                step=1
            )
            days = float(days_int)
            period_label = f"{days_int}일"

        st.sidebar.info(
            f"ℹ️ {timeframe} 봉: 호출 한도 총 {max_candles}개 캔들(≈ 최대 {total_max_days:.2f}일)."
        )
    else:
        max_days_int = int(total_max_days)
        days = st.sidebar.slider(
            f"백테스팅 기간 (일) - 최대 {max_days_int}일",
            min_value=7,
            max_value=max_days_int,
            value=min(30, max_days_int),
            step=1
        )
        period_label = f"{days}일"
        st.sidebar.info(
            f"ℹ️ {timeframe} 봉: 1회 최대 {max_candles_per_call}개 × {max_calls}회 = 총 {max_candles}개, 최대 {max_days_int}일"
        )
    
    # 백테스팅 설정
    st.sidebar.header("💰 백테스팅 설정")
    initial_capital = st.sidebar.number_input(
        "초기 자본 (USDT)",
        min_value=1000,
        max_value=100000,
        value=10000,
        step=1000
    )
    
    trade_ratio = st.sidebar.slider(
        "거래 비율",
        min_value=0.1,
        max_value=1.0,
        value=1.0,
        step=0.1,
        help="각 거래에서 사용할 자본의 비율"
    )
    leverage = st.sidebar.slider(
        "레버리지 (배)",
        min_value=1,
        max_value=20,
        value=1,
        step=1,
        help="레버리지 1배는 현물과 동일. 청산가는 대략 entry*(1-1/배수)로 계산(단순화된 롱 전용)"
    )
    
    # 지표 선택 섹션
    st.sidebar.header("📈 지표 선택")
    
    # RSI 설정
    use_rsi = st.sidebar.checkbox("RSI 사용", value=True)
    if use_rsi:
        with st.sidebar.expander("RSI 설정"):
            rsi_period = st.slider("RSI 기간", 5, 30, 14)
            rsi_oversold = st.slider("과매도 기준", 10, 40, 30)
            rsi_overbought = st.slider("과매수 기준", 60, 90, 70)
    
    # MACD 설정
    use_macd = st.sidebar.checkbox("MACD 사용", value=True)
    if use_macd:
        with st.sidebar.expander("MACD 설정"):
            macd_fast = st.slider("빠른 EMA", 5, 20, 12)
            macd_slow = st.slider("느린 EMA", 20, 40, 26)
            macd_signal = st.slider("신호선", 5, 15, 9)
    
    # 볼린저 밴드 설정
    use_bollinger = st.sidebar.checkbox("볼린저 밴드 사용", value=True)
    if use_bollinger:
        with st.sidebar.expander("볼린저 밴드 설정"):
            bb_period = st.slider("기간", 10, 30, 20)
            bb_std = st.slider("표준편차", 1.0, 3.0, 2.0, 0.1)
    
    # 이동평균선 설정
    use_ma = st.sidebar.checkbox("이동평균선 사용", value=True)
    if use_ma:
        with st.sidebar.expander("이동평균선 설정"):
            ma_type = st.selectbox("타입", ["sma", "ema"], 0)
            ma_short = st.slider("단기 기간", 5, 30, 20)
            ma_long = st.slider("장기 기간", 30, 100, 50)
    
    # 복합 전략 설정
    st.sidebar.header("🎯 복합 전략 설정")
    strategy_logic = st.sidebar.selectbox(
        "전략 로직",
        options=["any", "all"],
        index=0,
        help="any: 하나 이상의 지표 신호, all: 모든 지표 신호"
    )
    
    if strategy_logic == "any":
        min_signals = st.sidebar.slider(
            "최소 신호 개수",
            min_value=1,
            max_value=4,
            value=2,
            help="최소 몇 개의 지표가 같은 신호를 줘야 거래할지"
        )
    else:
        min_signals = 4  # all 모드에서는 모든 신호 필요
    
    # AI 전략 추천 (Gemini)
    st.sidebar.header("🤖 AI 전략 추천")
    with st.sidebar.expander("Gemini로 추천받기", expanded=False):
        asset_ko = st.selectbox("자산", ["비트", "이더", "솔라나", "에이다"], index=0)
        tf_ko = st.selectbox("시간 주기", ["1분", "3분", "5분", "15분", "30분", "1시간", "4시간", "1일"], index=5)
        profile = st.selectbox("성향", ["공격적", "안정적", "수비적"], index=1)

        def _map_asset(a: str) -> str:
            return {"비트": "BTC", "이더": "ETH", "솔라나": "SOL", "에이다": "ADA"}.get(a, "BTC")

        def _map_tf(t: str) -> str:
            return {
                "1분": "1m", "3분": "3m", "5분": "5m", "15분": "15m", "30분": "30m",
                "1시간": "1h", "4시간": "4h", "1일": "1d"
            }.get(t, "1h")

        def _build_gemini_prompt(asset_token: str, tf_token: str, profile_ko: str) -> str:
            return f"""
너는 퀀트 전략 디자이너다.
나는 {asset_token}, {tf_token} 기준으로 {profile_ko} 성향의 전략을 원한다.
사용 가능한 지표와 허용 범위는 아래와 같다. 이 범위 안에서 숫자만 선택해 간결히 제시하라.

RSI: period {{7–28}}, buy {{20–45}}, sell {{60–85}}

MACD: fast {{5–20}}, slow {{20–40}}(fast < slow), signal {{5–15}}

볼린저밴드: period {{10–40}}, std {{1.0–3.0}} (소수 허용)

이동평균선: type {{SMA|EMA}}, short {{5–30}}, long {{30–200}}(short < long)

중요: 네 지표를 모두 쓸 필요는 없다. 자산/시간주기/성향에 맞게 적절한 지표만 선택하라(최소 1개, 최대 4개).
사용하지 않는 지표의 항목은 출력에서 생략하라.
출력 형식은 정확히 아래처럼만 작성하라

SYMBOL: {{자산}}, TF: {{시간주기}}, PROFILE: {{성향}}
RSI: period=__, buy=__, sell=__  (선택한 경우에만 포함)
MACD: fast=__, slow=__, signal=__ (선택한 경우에만 포함)
BB: period=__, std=__            (선택한 경우에만 포함)
MA: type=__, short=__, long=__   (선택한 경우에만 포함)

전략 설명 : (전략에 대한 설명 첨부)
""".strip()

        def _call_gemini(prompt_text: str) -> str:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                return "[오류] GEMINI_API_KEY가 설정되지 않았습니다. .env에 GEMINI_API_KEY를 추가하세요."
            url = f"https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key={api_key}"
            payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
            try:
                r = requests.post(url, json=payload, timeout=30)
                j = r.json()
                if r.status_code != 200:
                    return f"[오류] {r.status_code} {j}"
                text = j.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                return text or "[오류] 응답에 텍스트가 없습니다."
            except Exception as e:
                return f"[오류] 요청 실패: {e}"

        if st.button("AI에게 추천받기", use_container_width=True):
            asset_token = _map_asset(asset_ko)
            tf_token = _map_tf(tf_ko)
            prompt_text = _build_gemini_prompt(asset_token, tf_token, profile)
            st.session_state["ai_prompt"] = prompt_text
            st.session_state["ai_response"] = _call_gemini(prompt_text)
            st.success("✅ AI 추천을 받았습니다. 본문에서 확인하세요.")

    # 백테스팅 실행 버튼
    if st.sidebar.button("🚀 백테스팅 실행", type="primary"):
        
        # 설정 딕셔너리 생성
        indicator_configs = {
            'use_rsi': use_rsi,
            'rsi_period': rsi_period if use_rsi else 14,
            'rsi_oversold': rsi_oversold if use_rsi else 30,
            'rsi_overbought': rsi_overbought if use_rsi else 70,
            'use_macd': use_macd,
            'macd_fast': macd_fast if use_macd else 12,
            'macd_slow': macd_slow if use_macd else 26,
            'macd_signal': macd_signal if use_macd else 9,
            'use_bollinger': use_bollinger,
            'bb_period': bb_period if use_bollinger else 20,
            'bb_std': bb_std if use_bollinger else 2,
            'use_ma': use_ma,
            'ma_type': ma_type if use_ma else 'sma',
            'ma_short': ma_short if use_ma else 20,
            'ma_long': ma_long if use_ma else 50
        }
        
        strategy_config = {
            'use_rsi': use_rsi,
            'use_macd': use_macd,
            'use_bollinger': use_bollinger,
            'use_ma': use_ma,
            'logic': strategy_logic,
            'min_signals': min_signals
        }
        
        # 백테스팅 실행
        with st.spinner("🔄 데이터 수집 및 백테스팅 실행 중..."):
            
            # 백테스팅 엔진 초기화
            engine = IntegratedBacktestingEngine()
            
            try:
                # 1. 데이터 수집
                st.info(f"📊 (선물) {ccxt_symbol} {timeframe} 데이터 수집 중... ({period_label})")
                df = engine.data_collector.fetch_historical_data_extended(ccxt_symbol, timeframe, days)
                
                if df is None or len(df) == 0:
                    st.error("❌ 데이터 수집에 실패했습니다.")
                    return
                
                # 2. 지표 계산
                st.info("📈 선택된 지표들 계산 중...")
                df_with_indicators = engine.calculate_all_indicators(df, indicator_configs)
                
                # 3. 복합 신호 생성
                st.info("🎯 복합 전략 신호 생성 중...")
                df_with_signals = engine.generate_combined_signals(df_with_indicators, strategy_config)
                
                # 4. 백테스팅 실행
                st.info("🚀 백테스팅 실행 중...")
                df_backtest, performance, trades = engine.run_backtest(
                    df_with_signals, initial_capital, trade_ratio, float(leverage)
                )
                
                # 5. 결과 저장 (세션 상태에)
                st.session_state.backtest_results = {
                    'df': df_backtest,
                    'performance': performance,
                    'trades': trades,
                    'configs': {
                        'symbol': symbol,
                        'ccxt_symbol': ccxt_symbol,
                        'tv_symbol': tradingview_symbol,
                        'timeframe': timeframe,
                        'days': days,
                        'indicator_configs': indicator_configs,
                        'strategy_config': strategy_config
                    }
                }
                
                # 백테스트 직후 기본 차트 선택을 '백테스팅 차트'로 유도
                st.session_state["chart_option"] = "백테스팅 차트"

                st.success("✅ 백테스팅 완료!")
                
            except Exception as e:
                st.error(f"❌ 백테스팅 실행 중 오류: {str(e)}")
                return
    
    # 결과 표시
    if 'backtest_results' in st.session_state:
        results = st.session_state.backtest_results
        df_result = results['df']
        performance = results['performance']
        trades = results['trades']
        configs = results['configs']
        
        # 호출된 기간 표기
        try:
            start_ts = df_result.index[0]
            end_ts = df_result.index[-1]
            period_days = (end_ts - start_ts).days
            st.info(
                f"🗓️ 호출된 기간: {start_ts.strftime('%Y-%m-%d %H:%M')} ~ {end_ts.strftime('%Y-%m-%d %H:%M')} (총 {period_days}일)"
            )
        except Exception:
            pass
        
        # 성과 지표 표시
        st.header("📊 백테스팅 결과")
        
        # 메트릭 표시
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "총 거래 수",
                f"{performance.get('total_trades', 0)}회",
                help="매수-매도 완성된 거래 쌍의 수"
            )
        
        with col2:
            total_return_pct = performance.get('total_return_pct', 0)
            st.metric(
                "총 수익률",
                f"{total_return_pct:.2f}%",
                delta=f"{total_return_pct:.2f}%"
            )
        
        with col3:
            win_rate = performance.get('win_rate', 0)
            st.metric(
                "승률",
                f"{win_rate:.1f}%",
                help="전체 거래 중 수익을 낸 거래의 비율"
            )
        
        with col4:
            final_value = performance.get('final_portfolio_value', initial_capital)
            delta_value = final_value - initial_capital
            st.metric(
                "최종 자산",
                f"${final_value:,.2f}",
                delta=delta_value,
                delta_color="normal"
            )
        
        # 상세 성과 지표
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 수익성 지표")
            st.write(f"**평균 거래 수익:** ${performance.get('avg_profit', 0):,.2f}")
            st.write(f"**최대 수익:** ${performance.get('max_profit', 0):,.2f}")
            st.write(f"**최대 손실:** ${performance.get('max_loss', 0):,.2f}")
        
        with col2:
            st.subheader("⏱️ 거래 빈도")
            st.write(f"**월평균 거래:** {performance.get('trades_per_month', 0):.1f}회")
            st.write(f"**평균 보유기간:** {performance.get('avg_holding_period', 0):.1f}일")
        
        # 차트 표시
        st.header("📈 백테스팅 차트")
        
        # 신호 데이터프레임을 먼저 준비 (어디에서든 참조 가능하도록)
        if 'Combined_Buy_Signal' in df_result.columns:
            buy_signals = df_result[df_result['Combined_Buy_Signal'] == True]
        else:
            buy_signals = pd.DataFrame()
        if 'Combined_Sell_Signal' in df_result.columns:
            sell_signals = df_result[df_result['Combined_Sell_Signal'] == True]
        else:
            sell_signals = pd.DataFrame()
        
        # 차트 옵션 선택
        col1, col2 = st.columns([3, 1])
        
        with col1:
            chart_option = st.radio(
                "차트 표시 방식",
                options=["백테스팅 차트", "TradingView 차트", "둘 다 표시"],
                index=0,
                key="chart_option",
                horizontal=True
            )
        
        with col2:
            chart_height = st.selectbox(
                "TradingView 차트 크기",
                options=[500, 600, 700, 800, 900],
                index=3,  # 800px 기본값
                help="TradingView 차트의 높이를 조절하세요"
            )
        
        # TradingView 차트 표시
        if chart_option in ["TradingView 차트", "둘 다 표시"]:
            st.subheader("📊 TradingView 실시간 차트")
            
            # TradingView 위젯 HTML (개선된 버전)
            tradingview_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="utf-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <style>
                    body {{
                        margin: 0;
                        padding: 0;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                        background-color: #ffffff;
                    }}
                    .tradingview-widget-container {{
                        height: {chart_height}px !important;
                        width: 100% !important;
                        position: relative;
                        box-sizing: border-box;
                        overflow: hidden;
                    }}
                    .tradingview-widget-container__widget {{
                        height: calc(100% - 32px) !important;
                        width: 100% !important;
                        position: relative;
                    }}
                    .tradingview-widget-copyright {{
                        font-size: 13px !important;
                        line-height: 32px !important;
                        text-align: center !important;
                        vertical-align: middle !important;
                        color: #9DB2BD !important;
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
                        background-color: #f8f9fa !important;
                        border-top: 1px solid #e9ecef !important;
                    }}
                    .tradingview-widget-copyright a {{
                        color: #2196F3 !important;
                        text-decoration: none !important;
                    }}
                    .tradingview-widget-copyright a:hover {{
                        text-decoration: underline !important;
                    }}
                </style>
            </head>
            <body>
                <div class="tradingview-widget-container">
                  <div class="tradingview-widget-container__widget"></div>
                  <div class="tradingview-widget-copyright">
                    <a href="https://kr.tradingview.com/" rel="noopener nofollow" target="_blank">
                      <span class="blue-text">TradingView에서 보기</span>
                    </a>
                  </div>
                  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
                  {{
                    "width": "100%",
                    "height": {chart_height - 32},
                    "symbol": "{configs.get('tv_symbol', 'BITGET:' + configs['symbol'])}",
                    "interval": "{configs['timeframe']}",
                    "timezone": "Asia/Seoul",
                    "theme": "light",
                    "style": "1",
                    "locale": "kr",
                    "enable_publishing": false,
                    "withdateranges": true,
                    "range": "3M",
                    "hide_side_toolbar": false,
                    "allow_symbol_change": true,
                    "details": true,
                    "hotlist": true,
                    "calendar": false,
                    "studies": [
                      "RSI@tv-basicstudies",
                      "MACD@tv-basicstudies", 
                      "BB@tv-basicstudies",
                      "MASimple@tv-basicstudies"
                    ],
                    "show_popup_button": true,
                    "popup_width": "1000",
                    "popup_height": "650",
                    "support_host": "https://www.tradingview.com",
                    "container_id": "tradingview_chart"
                  }}
                  </script>
                </div>
            </body>
            </html>
            """
            
            # HTML 컴포넌트로 표시 (전용 컨테이너로 폭 축소 이슈 방지)
            tv_full = st.container()
            with tv_full:
                st.components.v1.html(tradingview_html, height=chart_height)
            
            # 백테스팅 신호 요약 표시
            if len(buy_signals) > 0 or len(sell_signals) > 0:
                st.info(f"💡 **백테스팅 신호 요약**: 매수 {len(buy_signals)}회, 매도 {len(sell_signals)}회")
        
        # 백테스팅 차트 표시
        if chart_option in ["백테스팅 차트", "둘 다 표시"]:
            if chart_option == "둘 다 표시":
                st.subheader("🔍 백테스팅 상세 차트")
            
            # 메인 차트 생성
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=(
                    f'{configs["symbol"]} 가격 & 신호 ({configs["timeframe"]})',
                    '포트폴리오 가치'
                ),
                row_heights=[0.7, 0.3]
            )
            
            # 캔들스틱 차트
            fig.add_trace(
                go.Candlestick(
                    x=df_result.index,
                    open=df_result['open'],
                    high=df_result['high'],
                    low=df_result['low'],
                    close=df_result['close'],
                    name='Price',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ),
                row=1, col=1
            )
            
            # 매수/매도 신호 표시
            buy_signals = df_result[df_result['Combined_Buy_Signal'] == True]
            sell_signals = df_result[df_result['Combined_Sell_Signal'] == True]
            
            if len(buy_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=buy_signals.index,
                        y=buy_signals['close'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=15,
                            color='#00ff00',
                            line=dict(color='#008000', width=2)
                        ),
                        name='복합 매수 신호',
                        hovertemplate='<b>복합 매수 신호</b><br>' +
                                      '시간: %{x}<br>' +
                                      '가격: $%{y:,.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            if len(sell_signals) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=sell_signals.index,
                        y=sell_signals['close'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=15,
                            color='#ff0000',
                            line=dict(color='#800000', width=2)
                        ),
                        name='복합 매도 신호',
                        hovertemplate='<b>복합 매도 신호</b><br>' +
                                      '시간: %{x}<br>' +
                                      '가격: $%{y:,.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # 포트폴리오 가치 차트
            fig.add_trace(
                go.Scatter(
                    x=df_result.index,
                    y=df_result['Portfolio_Value'],
                    mode='lines',
                    name='포트폴리오 가치',
                    line=dict(color='#2196f3', width=2),
                    fill='tonexty',
                    fillcolor='rgba(33, 150, 243, 0.1)'
                ),
                row=2, col=1
            )
            
            # 초기 자본 기준선
            fig.add_hline(
                y=initial_capital,
                line_dash="dash",
                line_color="gray",
                opacity=0.5,
                row=2, col=1
            )
            
            # 레이아웃 설정
            fig.update_layout(
                title=f'통합 백테스팅 결과 - {configs["symbol"]} ({configs["timeframe"]})',
                height=800,
                showlegend=True,
                template='plotly_white'
            )
            
            fig.update_xaxes(rangeslider_visible=False)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 거래 내역
        if trades:
            st.header("📋 거래 내역")
            
            trades_df = pd.DataFrame(trades)
            trades_df['date'] = pd.to_datetime(trades_df['date'])
            trades_df = trades_df.sort_values('date', ascending=False)
            
            st.dataframe(
                trades_df.style.format({
                    'price': '${:,.2f}',
                    'quantity': '{:.6f}',
                    'amount': '${:,.2f}'
                }),
                use_container_width=True
            )
        
        # 사용된 전략 요약
        st.header("🎯 사용된 전략 요약")
        
        strategy_summary = []
        if configs['strategy_config']['use_rsi']:
            strategy_summary.append(f"RSI({configs['indicator_configs']['rsi_period']})")
        if configs['strategy_config']['use_macd']:
            strategy_summary.append(f"MACD({configs['indicator_configs']['macd_fast']},{configs['indicator_configs']['macd_slow']},{configs['indicator_configs']['macd_signal']})")
        if configs['strategy_config']['use_bollinger']:
            strategy_summary.append(f"BB({configs['indicator_configs']['bb_period']},{configs['indicator_configs']['bb_std']})")
        if configs['strategy_config']['use_ma']:
            strategy_summary.append(f"{configs['indicator_configs']['ma_type'].upper()}({configs['indicator_configs']['ma_short']},{configs['indicator_configs']['ma_long']})")
        
        st.info(f"**사용된 지표:** {', '.join(strategy_summary)}")
        st.info(f"**전략 로직:** {configs['strategy_config']['logic']} (최소 {configs['strategy_config']['min_signals']}개 신호)")

        # AI 추천 결과 표시
        if st.session_state.get("ai_prompt") or st.session_state.get("ai_response"):
            st.header("🤖 AI 전략 추천 결과 (Gemini)")
            with st.expander("전송 프롬프트 보기", expanded=False):
                st.code(st.session_state.get("ai_prompt", ""), language="text")
            st.subheader("추천 내용")
            st.code(st.session_state.get("ai_response", ""), language="text")
    
    else:
        # 백테스팅 결과가 없을 때 TradingView 차트만 표시
        st.header("📊 실시간 차트 미리보기")
        st.info("🚀 **백테스팅을 실행하면 더 자세한 분석을 볼 수 있습니다!**")
        
        # 차트 크기 선택
        preview_height = st.selectbox(
            "차트 크기",
            options=[500, 600, 700, 800],
            index=2,  # 700px 기본값
            help="미리보기 차트의 높이를 조절하세요"
        )
        
        # 기본 TradingView 차트 (현재 선택된 심볼과 시간프레임)
        default_tradingview = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background-color: #ffffff;
                }}
                .tradingview-widget-container {{
                    height: {preview_height}px !important;
                    width: 100% !important;
                    position: relative;
                    box-sizing: border-box;
                    overflow: hidden;
                }}
                .tradingview-widget-container__widget {{
                    height: calc(100% - 32px) !important;
                    width: 100% !important;
                    position: relative;
                }}
                .tradingview-widget-copyright {{
                    font-size: 13px !important;
                    line-height: 32px !important;
                    text-align: center !important;
                    vertical-align: middle !important;
                    color: #9DB2BD !important;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
                    background-color: #f8f9fa !important;
                    border-top: 1px solid #e9ecef !important;
                }}
                .tradingview-widget-copyright a {{
                    color: #2196F3 !important;
                    text-decoration: none !important;
                }}
                .tradingview-widget-copyright a:hover {{
                    text-decoration: underline !important;
                }}
            </style>
        </head>
        <body>
            <div class="tradingview-widget-container">
              <div class="tradingview-widget-container__widget"></div>
              <div class="tradingview-widget-copyright">
                <a href="https://kr.tradingview.com/" rel="noopener nofollow" target="_blank">
                  <span class="blue-text">TradingView에서 보기</span>
                </a>
              </div>
              <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
              {{
                "width": "100%",
                "height": {preview_height - 32},
                "symbol": "BITGET:{symbol}",
                "interval": "{timeframe}",
                "timezone": "Asia/Seoul",
                "theme": "light",
                "style": "1",
                "locale": "kr",
                "enable_publishing": false,
                "withdateranges": true,
                "range": "1M",
                "hide_side_toolbar": false,
                "allow_symbol_change": true,
                "details": true,
                "hotlist": true,
                "calendar": false,
                "studies": [
                  "RSI@tv-basicstudies",
                  "MACD@tv-basicstudies",
                  "BB@tv-basicstudies",
                  "MASimple@tv-basicstudies"
                ],
                "show_popup_button": true,
                "popup_width": "1000",
                "popup_height": "650",
                "support_host": "https://www.tradingview.com",
                "container_id": "tradingview_preview"
              }}
              </script>
            </div>
        </body>
        </html>
        """
        
        st.components.v1.html(default_tradingview, height=preview_height)
        
        # AI 추천 결과가 있으면 백테스트 전에도 표시
        if st.session_state.get("ai_prompt") or st.session_state.get("ai_response"):
            st.header("🤖 AI 전략 추천 결과 (Gemini)")
            with st.expander("전송 프롬프트 보기", expanded=False):
                st.code(st.session_state.get("ai_prompt", ""), language="text")
            st.subheader("추천 내용")
            st.code(st.session_state.get("ai_response", ""), language="text")

        # TradingView 차트 사용법
        with st.expander("📖 TradingView 차트 사용법"):
            st.write("""
            **🎯 TradingView 차트에서 할 수 있는 것들:**
            - 🔍 **확대/축소**: 마우스 휠 또는 + - 버튼
            - 📅 **시간 범위 변경**: 상단의 1D, 1W, 1M 버튼  
            - 📊 **지표 추가/제거**: 상단의 지표 버튼 (RSI, MACD 등이 이미 추가됨)
            - 🎨 **테마 변경**: 설정에서 다크/라이트 테마 변경
            - 💾 **차트 저장**: 우상단 카메라 아이콘으로 스크린샷
            - 🔄 **심볼 변경**: 좌상단에서 다른 코인으로 변경 가능
            - 📈 **전체화면**: 팝업 버튼으로 큰 화면에서 보기
            """)
        
        # 백테스팅 안내
        st.header("🚀 백테스팅 시작하기")
        st.divider()

        # 가독성을 위한 카드/칩 컴포넌트
        def render_card(title: str, desc: str, emoji: str = "", border: str = "#dbe4ff", bg: str = "#ffffff"):
            st.markdown(
                f"""
                <div style="border:1px solid {border}; background:{bg}; border-radius:10px; padding:12px; margin-bottom:10px;">
                    <div style="font-weight:700; font-size:15px; color:#0b1220;">
                        {emoji} {title}
                    </div>
                    <div style="color:#334155; font-size:13.5px; margin-top:6px; line-height:1.6;">
                        {desc}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        def render_chip(text: str, color: str = "#2563eb", bg: str = "#e8f0ff"):
            st.markdown(
                f"""
                <span style="display:inline-block; padding:4px 10px; border-radius:999px; font-weight:700; font-size:12.5px; color:{color}; background:{bg}; border:1px solid rgba(37, 99, 235, 0.15); margin-right:6px;">{text}</span>
                """,
                unsafe_allow_html=True,
            )

        tab1, tab2 = st.tabs(["📈 단일 지표 전략", "📊 복합 지표 전략 추천"])

        with tab1:
            c1, c2 = st.columns(2)
            with c1:
                render_card(
                    "RSI 전략",
                    "주가가 너무 올랐는지(과매수) 혹은 너무 떨어졌는지(과매도)를 보고 매매합니다.",
                    "📉",
                )
                render_card(
                    "볼린저밴드 전략",
                    "가격이 상단을 넘거나 하단을 깨면 반전 또는 돌파를 노립니다.",
                    "📊",
                )
            with c2:
                render_card(
                    "MACD 전략",
                    "골든·데드크로스로 추세가 위/아래로 바뀌는 시점을 포착합니다.",
                    "🔀",
                )
                render_card(
                    "이동평균선 전략",
                    "단기선과 장기선의 교차로 추세 전환 시 진입·청산합니다.",
                    "📈",
                )

        with tab2:
            with st.expander("RSI + 이동평균선 (추세 속 진입 타이밍 포착)", expanded=True):
                cols = st.columns([1, 1, 6])
                with cols[0]:
                    render_chip("RSI", color="#dc2626", bg="#fee2e2")
                with cols[1]:
                    render_chip("MA", color="#1d4ed8", bg="#dbeafe")
                st.markdown("- **추천 이유**: 이동평균선으로 큰 추세를 확인하고, RSI로 과매수·과매도 타이밍을 잡습니다.")
                st.markdown("- **설명**: 상승 추세일 때 RSI가 과매도 구간이면 매수, 하락 추세일 때 RSI가 과매수 구간이면 매도.")

            with st.expander("MACD + 볼린저밴드 (추세+변동성 돌파 매매)"):
                cols = st.columns([1, 1, 6])
                with cols[0]:
                    render_chip("MACD", color="#16a34a", bg="#dcfce7")
                with cols[1]:
                    render_chip("BB", color="#9333ea", bg="#f3e8ff")
                st.markdown("- **추천 이유**: MACD로 추세 전환을 확인하고, 볼린저밴드 돌파 시 강한 변동성을 노립니다.")
                st.markdown("- **설명**: MACD 골든크로스 시 상단 밴드 돌파는 강세 신호, 데드크로스 시 하단 밴드 이탈은 약세 신호.")

            with st.expander("이동평균선 + 볼린저밴드 (추세 필터링된 변동성 매매)"):
                cols = st.columns([1, 1, 6])
                with cols[0]:
                    render_chip("MA", color="#1d4ed8", bg="#dbeafe")
                with cols[1]:
                    render_chip("BB", color="#9333ea", bg="#f3e8ff")
                st.markdown("- **추천 이유**: 추세 방향과 변동성을 함께 고려해 가짜 신호를 줄입니다.")
                st.markdown("- **설명**: 이동평균선 위에서 상단 밴드 돌파 시 매수, 아래에서 하단 밴드 이탈 시 매도.")

            with st.expander("RSI + MACD (모멘텀과 추세 전환 결합)"):
                cols = st.columns([1, 1, 6])
                with cols[0]:
                    render_chip("RSI", color="#dc2626", bg="#fee2e2")
                with cols[1]:
                    render_chip("MACD", color="#16a34a", bg="#dcfce7")
                st.markdown("- **추천 이유**: RSI로 가격 과열 여부를 확인하고, MACD로 추세 전환 타이밍을 보강합니다.")
                st.markdown("- **설명**: RSI가 과매도 구간일 때 MACD 골든크로스가 발생하면 매수, 과매수 구간에서 데드크로스 발생 시 매도.")
        
        st.info("💡 **팁**: 왼쪽 사이드바에서 지표를 선택하고 '백테스팅 실행' 버튼을 누르세요!")


if __name__ == "__main__":
    create_integrated_backtesting_ui()