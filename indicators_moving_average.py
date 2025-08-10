"""
Moving Average (이동평균선) 지표 계산 및 분석 모듈

이동평균선은 가격의 추세를 파악하는 가장 기본적인 기술적 지표입니다.
- SMA (Simple Moving Average): 단순이동평균
- EMA (Exponential Moving Average): 지수이동평균
- Golden Cross / Death Cross: 단기/장기 이동평균선 교차 신호
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pandas_ta as ta


class MovingAverageIndicator:
    """이동평균선 지표 계산 및 신호 생성 클래스"""
    
    def __init__(self, short_period=20, long_period=50, ma_type='sma'):
        """
        이동평균선 지표 초기화
        
        Args:
            short_period (int): 단기 이동평균 기간 (기본값: 20)
            long_period (int): 장기 이동평균 기간 (기본값: 50)
            ma_type (str): 이동평균 타입 ('sma' 또는 'ema', 기본값: 'sma')
        """
        self.short_period = short_period
        self.long_period = long_period
        self.ma_type = ma_type.lower()
        
        if self.ma_type not in ['sma', 'ema']:
            raise ValueError("ma_type은 'sma' 또는 'ema'여야 합니다.")
    
    def calculate_moving_averages(self, df):
        """
        이동평균선 계산
        
        Args:
            df (pd.DataFrame): OHLCV 데이터 (close 컬럼 필요)
            
        Returns:
            pd.DataFrame: 이동평균선 컬럼이 추가된 데이터프레임
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame에 'close' 컬럼이 필요합니다.")
        
        df_result = df.copy()
        
        if self.ma_type == 'sma':
            # 단순이동평균 (SMA) 계산
            df_result[f'SMA_{self.short_period}'] = ta.sma(df['close'], length=self.short_period)
            df_result[f'SMA_{self.long_period}'] = ta.sma(df['close'], length=self.long_period)
            
            # 별칭 생성
            df_result['MA_Short'] = df_result[f'SMA_{self.short_period}']
            df_result['MA_Long'] = df_result[f'SMA_{self.long_period}']
            
        else:  # ema
            # 지수이동평균 (EMA) 계산
            df_result[f'EMA_{self.short_period}'] = ta.ema(df['close'], length=self.short_period)
            df_result[f'EMA_{self.long_period}'] = ta.ema(df['close'], length=self.long_period)
            
            # 별칭 생성
            df_result['MA_Short'] = df_result[f'EMA_{self.short_period}']
            df_result['MA_Long'] = df_result[f'EMA_{self.long_period}']
        
        return df_result
    
    def analyze_ma_signals(self, df):
        """
        이동평균선 매매 신호 분석
        
        Args:
            df (pd.DataFrame): 이동평균선이 계산된 데이터프레임
            
        Returns:
            pd.DataFrame: 매매 신호가 추가된 데이터프레임
        """
        df_result = df.copy()
        
        # 매매 신호 초기화
        df_result['MA_Signal_Type'] = 'hold'
        df_result['MA_Buy_Signal'] = False
        df_result['MA_Sell_Signal'] = False
        
        # 이동평균선 교차 신호
        # Golden Cross: 단기 이동평균이 장기 이동평균을 상향 돌파 (매수 신호)
        # Death Cross: 단기 이동평균이 장기 이동평균을 하향 돌파 (매도 신호)
        
        ma_short_prev = df_result['MA_Short'].shift(1)
        ma_long_prev = df_result['MA_Long'].shift(1)
        
        # Golden Cross (매수 신호)
        golden_cross = (
            (df_result['MA_Short'] > df_result['MA_Long']) &
            (ma_short_prev <= ma_long_prev)
        )
        
        # Death Cross (매도 신호)
        death_cross = (
            (df_result['MA_Short'] < df_result['MA_Long']) &
            (ma_short_prev >= ma_long_prev)
        )
        
        # 추가 조건: 가격이 이동평균선 위/아래에 있는지 확인
        price_above_short = df_result['close'] > df_result['MA_Short']
        price_below_short = df_result['close'] < df_result['MA_Short']
        
        # 매수 신호: Golden Cross + 가격이 단기 이동평균 위에 있음
        buy_condition = golden_cross & price_above_short
        
        # 매도 신호: Death Cross + 가격이 단기 이동평균 아래에 있음
        sell_condition = death_cross & price_below_short
        
        df_result.loc[buy_condition, 'MA_Signal_Type'] = 'buy'
        df_result.loc[buy_condition, 'MA_Buy_Signal'] = True
        
        df_result.loc[sell_condition, 'MA_Signal_Type'] = 'sell'
        df_result.loc[sell_condition, 'MA_Sell_Signal'] = True
        
        return df_result
    
    def get_ma_summary(self, df):
        """
        이동평균선 지표 요약 정보
        
        Args:
            df (pd.DataFrame): 이동평균선이 계산된 데이터프레임
            
        Returns:
            dict: 이동평균선 요약 정보
        """
        valid_ma = df[['MA_Short', 'MA_Long']].dropna()
        
        buy_signals = df['MA_Buy_Signal'].sum()
        sell_signals = df['MA_Sell_Signal'].sum()
        
        if len(valid_ma) > 0:
            current_price = df['close'].iloc[-1]
            current_short = valid_ma['MA_Short'].iloc[-1]
            current_long = valid_ma['MA_Long'].iloc[-1]
            
            # 현재 추세 판단
            if current_short > current_long:
                if current_price > current_short:
                    trend = "강한 상승 추세 (가격 > 단기 > 장기)"
                else:
                    trend = "약한 상승 추세 (단기 > 장기 > 가격)"
            else:
                if current_price < current_short:
                    trend = "강한 하락 추세 (가격 < 단기 < 장기)"
                else:
                    trend = "약한 하락 추세 (단기 < 장기 < 가격)"
            
            # 가격과 이동평균선 거리 (%)
            short_distance = ((current_price - current_short) / current_short) * 100
            long_distance = ((current_price - current_long) / current_long) * 100
            
        else:
            current_price = current_short = current_long = None
            trend = "데이터 부족"
            short_distance = long_distance = 0
        
        return {
            'ma_type': self.ma_type.upper(),
            'short_period': self.short_period,
            'long_period': self.long_period,
            'valid_values': len(valid_ma),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'total_signals': buy_signals + sell_signals,
            'current_price': current_price,
            'current_short': current_short,
            'current_long': current_long,
            'trend': trend,
            'short_distance_pct': short_distance,
            'long_distance_pct': long_distance,
            'short_range': f"{valid_ma['MA_Short'].min():.2f} ~ {valid_ma['MA_Short'].max():.2f}" if len(valid_ma) > 0 else "N/A",
            'long_range': f"{valid_ma['MA_Long'].min():.2f} ~ {valid_ma['MA_Long'].max():.2f}" if len(valid_ma) > 0 else "N/A"
        }
    
    def create_ma_chart(self, df, symbol="BTCUSDT", timeframe="1h"):
        """
        이동평균선 차트 생성
        
        Args:
            df (pd.DataFrame): 이동평균선이 계산된 데이터프레임
            symbol (str): 거래 심볼
            timeframe (str): 시간 간격
            
        Returns:
            plotly.graph_objects.Figure: 이동평균선 차트
        """
        fig = go.Figure()
        
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
            )
        )
        
        # 2. 단기 이동평균선
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA_Short'],
                mode='lines',
                name=f'{self.ma_type.upper()}{self.short_period}',
                line=dict(color='#2196f3', width=2),
                hovertemplate=f'<b>{self.ma_type.upper()}{self.short_period}</b><br>' +
                              '시간: %{x}<br>' +
                              '값: $%{y:,.2f}<extra></extra>'
            )
        )
        
        # 3. 장기 이동평균선
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['MA_Long'],
                mode='lines',
                name=f'{self.ma_type.upper()}{self.long_period}',
                line=dict(color='#ff9800', width=2),
                hovertemplate=f'<b>{self.ma_type.upper()}{self.long_period}</b><br>' +
                              '시간: %{x}<br>' +
                              '값: $%{y:,.2f}<extra></extra>'
            )
        )
        
        # 4. 매수/매도 신호 표시
        buy_signals = df[df['MA_Buy_Signal'] == True]
        sell_signals = df[df['MA_Sell_Signal'] == True]
        
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
                    name='Golden Cross',
                    hovertemplate='<b>Golden Cross (매수 신호)</b><br>' +
                                  '시간: %{x}<br>' +
                                  '가격: $%{y:,.2f}<extra></extra>'
                )
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
                    name='Death Cross',
                    hovertemplate='<b>Death Cross (매도 신호)</b><br>' +
                                  '시간: %{x}<br>' +
                                  '가격: $%{y:,.2f}<extra></extra>'
                )
            )
        
        # 5. 이동평균선 사이 영역 채우기 (추세 시각화)
        # 단기 > 장기일 때 상승 추세 (초록색)
        # 단기 < 장기일 때 하락 추세 (빨간색)
        for i in range(1, len(df)):
            if pd.notna(df['MA_Short'].iloc[i]) and pd.notna(df['MA_Long'].iloc[i]):
                if df['MA_Short'].iloc[i] > df['MA_Long'].iloc[i]:
                    color = 'rgba(76, 175, 80, 0.1)'  # 상승 추세 (연한 초록)
                else:
                    color = 'rgba(244, 67, 54, 0.1)'   # 하락 추세 (연한 빨강)
                
                fig.add_trace(
                    go.Scatter(
                        x=[df.index[i-1], df.index[i], df.index[i], df.index[i-1]],
                        y=[df['MA_Short'].iloc[i-1], df['MA_Short'].iloc[i], 
                           df['MA_Long'].iloc[i], df['MA_Long'].iloc[i-1]],
                        fill='toself',
                        fillcolor=color,
                        line=dict(color='rgba(255,255,255,0)'),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )
        
        # 레이아웃 설정
        fig.update_layout(
            title=f'{symbol} 이동평균선 분석 ({self.ma_type.upper()}{self.short_period}/{self.long_period}) - {timeframe}',
            xaxis_title='시간',
            yaxis_title='가격 ($)',
            height=600,
            showlegend=True,
            template='plotly_white',
            xaxis_rangeslider_visible=False
        )
        
        return fig


def test_moving_average_calculation():
    """이동평균선 계산 테스트"""
    print("🧪 이동평균선 계산 테스트")
    
    # 샘플 데이터 생성
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    np.random.seed(42)
    prices = 50000 + np.cumsum(np.random.randn(100) * 100)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(100, 1000, 100)
    }, index=dates)
    
    # SMA 테스트
    print("\n📊 SMA (단순이동평균) 테스트:")
    sma = MovingAverageIndicator(short_period=10, long_period=20, ma_type='sma')
    df_with_sma = sma.calculate_moving_averages(df)
    df_with_sma_signals = sma.analyze_ma_signals(df_with_sma)
    
    sma_summary = sma.get_ma_summary(df_with_sma_signals)
    print(f"   - 유효한 값: {sma_summary['valid_values']}개")
    print(f"   - 매수 신호: {sma_summary['buy_signals']}개")
    print(f"   - 매도 신호: {sma_summary['sell_signals']}개")
    print(f"   - 현재 추세: {sma_summary['trend']}")
    
    # EMA 테스트
    print("\n📊 EMA (지수이동평균) 테스트:")
    ema = MovingAverageIndicator(short_period=10, long_period=20, ma_type='ema')
    df_with_ema = ema.calculate_moving_averages(df)
    df_with_ema_signals = ema.analyze_ma_signals(df_with_ema)
    
    ema_summary = ema.get_ma_summary(df_with_ema_signals)
    print(f"   - 유효한 값: {ema_summary['valid_values']}개")
    print(f"   - 매수 신호: {ema_summary['buy_signals']}개")
    print(f"   - 매도 신호: {ema_summary['sell_signals']}개")
    print(f"   - 현재 추세: {ema_summary['trend']}")
    
    print("\n✅ 이동평균선 계산 완료!")


if __name__ == "__main__":
    test_moving_average_calculation()