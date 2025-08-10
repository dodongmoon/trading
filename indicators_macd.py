"""
MACD (Moving Average Convergence Divergence) 지표 계산 및 분석 모듈

MACD는 두 개의 지수이동평균(EMA) 간의 차이를 나타내는 모멘텀 지표입니다.
- MACD Line: 12일 EMA - 26일 EMA
- Signal Line: MACD의 9일 EMA
- Histogram: MACD Line - Signal Line
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pandas_ta as ta


class MACDIndicator:
    """MACD 지표 계산 및 신호 생성 클래스"""
    
    def __init__(self, fast_period=12, slow_period=26, signal_period=9):
        """
        MACD 지표 초기화
        
        Args:
            fast_period (int): 빠른 EMA 기간 (기본값: 12)
            slow_period (int): 느린 EMA 기간 (기본값: 26)
            signal_period (int): 신호선 EMA 기간 (기본값: 9)
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
    
    def calculate_macd(self, df):
        """
        MACD 지표 계산
        
        Args:
            df (pd.DataFrame): OHLCV 데이터 (close 컬럼 필요)
            
        Returns:
            pd.DataFrame: MACD, Signal, Histogram 컬럼이 추가된 데이터프레임
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame에 'close' 컬럼이 필요합니다.")
        
        # pandas_ta를 사용한 MACD 계산
        macd_data = ta.macd(
            df['close'], 
            fast=self.fast_period, 
            slow=self.slow_period, 
            signal=self.signal_period
        )
        
        # 결과를 원본 DataFrame에 추가
        df_result = df.copy()
        df_result[f'MACD_{self.fast_period}_{self.slow_period}_{self.signal_period}'] = macd_data[f'MACD_{self.fast_period}_{self.slow_period}_{self.signal_period}']
        df_result[f'MACDh_{self.fast_period}_{self.slow_period}_{self.signal_period}'] = macd_data[f'MACDh_{self.fast_period}_{self.slow_period}_{self.signal_period}']
        df_result[f'MACDs_{self.fast_period}_{self.slow_period}_{self.signal_period}'] = macd_data[f'MACDs_{self.fast_period}_{self.slow_period}_{self.signal_period}']
        
        # 간단한 컬럼명으로 별칭 생성
        df_result['MACD'] = df_result[f'MACD_{self.fast_period}_{self.slow_period}_{self.signal_period}']
        df_result['MACD_Signal'] = df_result[f'MACDs_{self.fast_period}_{self.slow_period}_{self.signal_period}']
        df_result['MACD_Histogram'] = df_result[f'MACDh_{self.fast_period}_{self.slow_period}_{self.signal_period}']
        
        return df_result
    
    def analyze_macd_signals(self, df):
        """
        MACD 매매 신호 분석
        
        Args:
            df (pd.DataFrame): MACD가 계산된 데이터프레임
            
        Returns:
            pd.DataFrame: 매매 신호가 추가된 데이터프레임
        """
        df_result = df.copy()
        
        # 매매 신호 초기화
        df_result['MACD_Signal_Type'] = 'hold'
        df_result['MACD_Buy_Signal'] = False
        df_result['MACD_Sell_Signal'] = False
        
        # MACD Line과 Signal Line 교차 신호
        # Golden Cross: MACD가 Signal을 상향 돌파 (매수 신호)
        # Death Cross: MACD가 Signal을 하향 돌파 (매도 신호)
        
        macd_prev = df_result['MACD'].shift(1)
        signal_prev = df_result['MACD_Signal'].shift(1)
        
        # 매수 신호: MACD가 Signal Line을 상향 돌파
        buy_condition = (
            (df_result['MACD'] > df_result['MACD_Signal']) &
            (macd_prev <= signal_prev)
        )
        
        # 매도 신호: MACD가 Signal Line을 하향 돌파  
        sell_condition = (
            (df_result['MACD'] < df_result['MACD_Signal']) &
            (macd_prev >= signal_prev)
        )
        
        df_result.loc[buy_condition, 'MACD_Signal_Type'] = 'buy'
        df_result.loc[buy_condition, 'MACD_Buy_Signal'] = True
        
        df_result.loc[sell_condition, 'MACD_Signal_Type'] = 'sell'
        df_result.loc[sell_condition, 'MACD_Sell_Signal'] = True
        
        return df_result
    
    def get_macd_summary(self, df):
        """
        MACD 지표 요약 정보
        
        Args:
            df (pd.DataFrame): MACD가 계산된 데이터프레임
            
        Returns:
            dict: MACD 요약 정보
        """
        valid_macd = df['MACD'].dropna()
        valid_signal = df['MACD_Signal'].dropna()
        valid_histogram = df['MACD_Histogram'].dropna()
        
        buy_signals = df['MACD_Buy_Signal'].sum()
        sell_signals = df['MACD_Sell_Signal'].sum()
        
        current_macd = valid_macd.iloc[-1] if len(valid_macd) > 0 else None
        current_signal = valid_signal.iloc[-1] if len(valid_signal) > 0 else None
        current_histogram = valid_histogram.iloc[-1] if len(valid_histogram) > 0 else None
        
        # 현재 상태 판단
        if current_macd is not None and current_signal is not None:
            if current_macd > current_signal:
                current_trend = "상승 (MACD > Signal)"
            else:
                current_trend = "하락 (MACD < Signal)"
        else:
            current_trend = "데이터 부족"
        
        return {
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'valid_values': len(valid_macd),
            'macd_range': f"{valid_macd.min():.4f} ~ {valid_macd.max():.4f}" if len(valid_macd) > 0 else "N/A",
            'signal_range': f"{valid_signal.min():.4f} ~ {valid_signal.max():.4f}" if len(valid_signal) > 0 else "N/A",
            'histogram_range': f"{valid_histogram.min():.4f} ~ {valid_histogram.max():.4f}" if len(valid_histogram) > 0 else "N/A",
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'total_signals': buy_signals + sell_signals,
            'current_macd': current_macd,
            'current_signal': current_signal,
            'current_histogram': current_histogram,
            'current_trend': current_trend
        }
    
    def create_macd_chart(self, df, symbol="BTCUSDT", timeframe="1h"):
        """
        MACD 지표 차트 생성
        
        Args:
            df (pd.DataFrame): MACD가 계산된 데이터프레임
            symbol (str): 거래 심볼
            timeframe (str): 시간 간격
            
        Returns:
            plotly.graph_objects.Figure: MACD 차트
        """
        # 서브플롯 생성 (가격 차트 + MACD)
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} 가격 차트 ({timeframe})', 'MACD'),
            row_width=[0.7, 0.3]
        )
        
        # 1. 캔들스틱 차트
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )
        
        # 2. 매수/매도 신호 표시
        buy_signals = df[df['MACD_Buy_Signal'] == True]
        sell_signals = df[df['MACD_Sell_Signal'] == True]
        
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_signals.index,
                    y=buy_signals['close'],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up',
                        size=12,
                        color='#00ff00',
                        line=dict(color='#008000', width=2)
                    ),
                    name='MACD 매수',
                    hovertemplate='<b>MACD 매수 신호</b><br>' +
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
                        size=12,
                        color='#ff0000',
                        line=dict(color='#800000', width=2)
                    ),
                    name='MACD 매도',
                    hovertemplate='<b>MACD 매도 신호</b><br>' +
                                  '시간: %{x}<br>' +
                                  '가격: $%{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 3. MACD Line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD'],
                mode='lines',
                name='MACD',
                line=dict(color='#2196f3', width=2),
                hovertemplate='<b>MACD</b><br>' +
                              '시간: %{x}<br>' +
                              '값: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 4. Signal Line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MACD_Signal'],
                mode='lines',
                name='Signal',
                line=dict(color='#ff9800', width=2),
                hovertemplate='<b>Signal Line</b><br>' +
                              '시간: %{x}<br>' +
                              '값: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 5. Histogram
        colors = ['#26a69a' if h >= 0 else '#ef5350' for h in df['MACD_Histogram']]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['MACD_Histogram'],
                name='Histogram',
                marker_color=colors,
                opacity=0.7,
                hovertemplate='<b>MACD Histogram</b><br>' +
                              '시간: %{x}<br>' +
                              '값: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 6. 제로 라인 추가
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
        
        # 레이아웃 설정
        fig.update_layout(
            title=f'{symbol} MACD 분석 ({timeframe})',
            xaxis_title='시간',
            yaxis_title='가격 ($)',
            xaxis2_title='시간',
            yaxis2_title='MACD 값',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # x축 범위 동기화
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig


def test_macd_calculation():
    """MACD 계산 테스트"""
    print("🧪 MACD 계산 테스트")
    
    # 샘플 데이터 생성
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(100) * 100)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(100, 1000, 100)
    }, index=dates)
    
    # MACD 계산
    macd = MACDIndicator()
    df_with_macd = macd.calculate_macd(df)
    df_with_signals = macd.analyze_macd_signals(df_with_macd)
    
    # 결과 출력
    summary = macd.get_macd_summary(df_with_signals)
    print(f"✅ MACD 계산 완료!")
    print(f"   - 유효한 값: {summary['valid_values']}개")
    print(f"   - MACD 범위: {summary['macd_range']}")
    print(f"   - 매수 신호: {summary['buy_signals']}개")
    print(f"   - 매도 신호: {summary['sell_signals']}개")
    print(f"   - 현재 추세: {summary['current_trend']}")


if __name__ == "__main__":
    test_macd_calculation()