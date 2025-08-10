"""
Bollinger Bands (볼린저 밴드) 지표 계산 및 분석 모듈

볼린저 밴드는 가격의 변동성을 측정하는 기술적 지표입니다.
- 중간선: 단순이동평균 (SMA)
- 상단밴드: 중간선 + (표준편차 × 승수)  
- 하단밴드: 중간선 - (표준편차 × 승수)
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import pandas_ta as ta


class BollingerBandsIndicator:
    """볼린저 밴드 지표 계산 및 신호 생성 클래스"""
    
    def __init__(self, period=20, std_dev=2):
        """
        볼린저 밴드 지표 초기화
        
        Args:
            period (int): 이동평균 기간 (기본값: 20)
            std_dev (float): 표준편차 승수 (기본값: 2)
        """
        self.period = period
        self.std_dev = std_dev
    
    def calculate_bollinger_bands(self, df):
        """
        볼린저 밴드 계산
        
        Args:
            df (pd.DataFrame): OHLCV 데이터 (close 컬럼 필요)
            
        Returns:
            pd.DataFrame: 볼린저 밴드 컬럼이 추가된 데이터프레임
        """
        if 'close' not in df.columns:
            raise ValueError("DataFrame에 'close' 컬럼이 필요합니다.")
        
        # pandas_ta를 사용한 볼린저 밴드 계산
        bb_data = ta.bbands(
            df['close'], 
            length=self.period, 
            std=self.std_dev
        )
        
        # 실제 컬럼명 확인을 위한 디버깅
        if bb_data is not None:
            print(f"📊 볼린저 밴드 컬럼들: {list(bb_data.columns)}")
        
        # 결과를 원본 DataFrame에 추가
        df_result = df.copy()
        
        # pandas_ta의 실제 컬럼명 사용
        for col in bb_data.columns:
            df_result[col] = bb_data[col]
        
        # 간단한 컬럼명으로 별칭 생성 (실제 컬럼명 사용)
        df_result['BB_Lower'] = df_result[f'BBL_{self.period}_{float(self.std_dev)}']
        df_result['BB_Middle'] = df_result[f'BBM_{self.period}_{float(self.std_dev)}']
        df_result['BB_Upper'] = df_result[f'BBU_{self.period}_{float(self.std_dev)}']
        df_result['BB_Percent'] = df_result[f'BBB_{self.period}_{float(self.std_dev)}']  # %B (0~1)
        df_result['BB_Width'] = df_result[f'BBP_{self.period}_{float(self.std_dev)}']   # Bandwidth
        
        return df_result
    
    def analyze_bollinger_signals(self, df):
        """
        볼린저 밴드 매매 신호 분석
        
        Args:
            df (pd.DataFrame): 볼린저 밴드가 계산된 데이터프레임
            
        Returns:
            pd.DataFrame: 매매 신호가 추가된 데이터프레임
        """
        df_result = df.copy()
        
        # 매매 신호 초기화
        df_result['BB_Signal_Type'] = 'hold'
        df_result['BB_Buy_Signal'] = False
        df_result['BB_Sell_Signal'] = False
        
        # 볼린저 밴드 매매 전략
        # 1. 하단밴드 터치 후 반등 (매수 신호)
        # 2. 상단밴드 터치 후 하락 (매도 신호)
        # 3. %B 지표 활용 (0.2 이하에서 매수, 0.8 이상에서 매도)
        
        close_prev = df_result['close'].shift(1)
        bb_percent = df_result['BB_Percent']
        
        # 매수 신호: 
        # - 가격이 하단밴드 근처에서 반등 (%B < 0.2에서 0.2 이상으로 상승)
        # - 또는 중간선을 상향 돌파
        buy_condition = (
            # %B가 0.2 이하에서 0.2 이상으로 상승 (과매도 구간에서 반등)
            ((bb_percent > 0.2) & (bb_percent.shift(1) <= 0.2)) |
            # 가격이 중간선을 상향 돌파
            ((df_result['close'] > df_result['BB_Middle']) & 
             (close_prev <= df_result['BB_Middle'].shift(1)))
        )
        
        # 매도 신호:
        # - 가격이 상단밴드 근처에서 하락 (%B > 0.8에서 0.8 이하로 하락)
        # - 또는 중간선을 하향 돌파
        sell_condition = (
            # %B가 0.8 이상에서 0.8 이하로 하락 (과매수 구간에서 하락)
            ((bb_percent < 0.8) & (bb_percent.shift(1) >= 0.8)) |
            # 가격이 중간선을 하향 돌파
            ((df_result['close'] < df_result['BB_Middle']) & 
             (close_prev >= df_result['BB_Middle'].shift(1)))
        )
        
        df_result.loc[buy_condition, 'BB_Signal_Type'] = 'buy'
        df_result.loc[buy_condition, 'BB_Buy_Signal'] = True
        
        df_result.loc[sell_condition, 'BB_Signal_Type'] = 'sell'
        df_result.loc[sell_condition, 'BB_Sell_Signal'] = True
        
        return df_result
    
    def get_bollinger_summary(self, df):
        """
        볼린저 밴드 지표 요약 정보
        
        Args:
            df (pd.DataFrame): 볼린저 밴드가 계산된 데이터프레임
            
        Returns:
            dict: 볼린저 밴드 요약 정보
        """
        valid_bb = df[['BB_Lower', 'BB_Middle', 'BB_Upper', 'BB_Percent', 'BB_Width']].dropna()
        
        buy_signals = df['BB_Buy_Signal'].sum()
        sell_signals = df['BB_Sell_Signal'].sum()
        
        if len(valid_bb) > 0:
            current_price = df['close'].iloc[-1]
            current_lower = valid_bb['BB_Lower'].iloc[-1]
            current_middle = valid_bb['BB_Middle'].iloc[-1]
            current_upper = valid_bb['BB_Upper'].iloc[-1]
            current_percent = valid_bb['BB_Percent'].iloc[-1]
            current_width = valid_bb['BB_Width'].iloc[-1]
            
            # 현재 위치 판단
            if current_percent <= 0.2:
                position = "과매도 구간 (하단밴드 근처)"
            elif current_percent >= 0.8:
                position = "과매수 구간 (상단밴드 근처)"
            elif 0.4 <= current_percent <= 0.6:
                position = "중립 구간 (중간선 근처)"
            elif current_percent < 0.4:
                position = "하락 구간"
            else:
                position = "상승 구간"
            
            # 밴드 폭 상태
            avg_width = valid_bb['BB_Width'].mean()
            if current_width > avg_width * 1.2:
                width_status = "확장 (변동성 증가)"
            elif current_width < avg_width * 0.8:
                width_status = "수축 (변동성 감소)"
            else:
                width_status = "보통"
        else:
            current_price = current_lower = current_middle = current_upper = None
            current_percent = current_width = None
            position = width_status = "데이터 부족"
        
        return {
            'period': self.period,
            'std_dev': self.std_dev,
            'valid_values': len(valid_bb),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'total_signals': buy_signals + sell_signals,
            'current_price': current_price,
            'current_lower': current_lower,
            'current_middle': current_middle,
            'current_upper': current_upper,
            'current_percent': current_percent,
            'current_width': current_width,
            'position': position,
            'width_status': width_status,
            'bb_range': f"{valid_bb['BB_Lower'].min():.2f} ~ {valid_bb['BB_Upper'].max():.2f}" if len(valid_bb) > 0 else "N/A"
        }
    
    def create_bollinger_chart(self, df, symbol="BTCUSDT", timeframe="1h"):
        """
        볼린저 밴드 차트 생성
        
        Args:
            df (pd.DataFrame): 볼린저 밴드가 계산된 데이터프레임
            symbol (str): 거래 심볼
            timeframe (str): 시간 간격
            
        Returns:
            plotly.graph_objects.Figure: 볼린저 밴드 차트
        """
        # 서브플롯 생성 (가격 차트 + %B 지표)
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} 볼린저 밴드 ({timeframe})', '%B 지표'),
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
        
        # 2. 볼린저 밴드 라인들
        # 상단 밴드
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Upper'],
                mode='lines',
                name='상단밴드',
                line=dict(color='#ff5722', width=1),
                hovertemplate='<b>상단밴드</b><br>시간: %{x}<br>값: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 중간선 (이동평균)
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Middle'],
                mode='lines',
                name='중간선 (SMA)',
                line=dict(color='#2196f3', width=2),
                hovertemplate='<b>중간선</b><br>시간: %{x}<br>값: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 하단 밴드
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Lower'],
                mode='lines',
                name='하단밴드',
                line=dict(color='#4caf50', width=1),
                hovertemplate='<b>하단밴드</b><br>시간: %{x}<br>값: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # 3. 밴드 사이 영역 채우기
        fig.add_trace(
            go.Scatter(
                x=df.index.tolist() + df.index.tolist()[::-1],
                y=df['BB_Upper'].tolist() + df['BB_Lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(33, 150, 243, 0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='볼린저 밴드',
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        # 4. 매수/매도 신호 표시
        buy_signals = df[df['BB_Buy_Signal'] == True]
        sell_signals = df[df['BB_Sell_Signal'] == True]
        
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
                    name='BB 매수',
                    hovertemplate='<b>볼린저 밴드 매수 신호</b><br>' +
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
                    name='BB 매도',
                    hovertemplate='<b>볼린저 밴드 매도 신호</b><br>' +
                                  '시간: %{x}<br>' +
                                  '가격: $%{y:,.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # 5. %B 지표
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['BB_Percent'],
                mode='lines',
                name='%B',
                line=dict(color='#9c27b0', width=2),
                hovertemplate='<b>%B 지표</b><br>' +
                              '시간: %{x}<br>' +
                              '값: %{y:.3f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # 6. %B 기준선들 (과매수/과매도 구간)
        fig.add_hline(y=0.8, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)
        fig.add_hline(y=0.2, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
        
        # 과매수/과매도 구간 색칠
        fig.add_hrect(y0=0.8, y1=1.0, fillcolor="red", opacity=0.1, row=2, col=1)
        fig.add_hrect(y0=0.0, y1=0.2, fillcolor="green", opacity=0.1, row=2, col=1)
        
        # 레이아웃 설정
        fig.update_layout(
            title=f'{symbol} 볼린저 밴드 분석 ({timeframe})',
            xaxis_title='시간',
            yaxis_title='가격 ($)',
            xaxis2_title='시간',
            yaxis2_title='%B 값',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        # x축 범위 동기화
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig


def test_bollinger_calculation():
    """볼린저 밴드 계산 테스트"""
    print("🧪 볼린저 밴드 계산 테스트")
    
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
    
    # 볼린저 밴드 계산
    bb = BollingerBandsIndicator()
    df_with_bb = bb.calculate_bollinger_bands(df)
    df_with_signals = bb.analyze_bollinger_signals(df_with_bb)
    
    # 결과 출력
    summary = bb.get_bollinger_summary(df_with_signals)
    print(f"✅ 볼린저 밴드 계산 완료!")
    print(f"   - 유효한 값: {summary['valid_values']}개")
    print(f"   - 밴드 범위: {summary['bb_range']}")
    print(f"   - 매수 신호: {summary['buy_signals']}개")
    print(f"   - 매도 신호: {summary['sell_signals']}개")
    print(f"   - 현재 위치: {summary['position']}")
    print(f"   - 밴드 폭 상태: {summary['width_status']}")


if __name__ == "__main__":
    test_bollinger_calculation()