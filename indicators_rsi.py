import pandas as pd
import pandas_ta as ta
import numpy as np
from data_collector import BitgetDataCollector
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class RSIIndicator:
    def __init__(self, period=14):
        """
        RSI 지표 계산기 초기화
        
        Args:
            period (int): RSI 계산 기간 (기본값: 14)
        """
        self.period = period
        
    def calculate_rsi(self, df):
        """
        RSI 지표를 계산합니다.
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            
        Returns:
            pd.DataFrame: RSI가 추가된 DataFrame
        """
        df_copy = df.copy()
        
        # pandas_ta를 사용한 RSI 계산
        df_copy[f'RSI_{self.period}'] = ta.rsi(df_copy['close'], length=self.period)
        
        print(f"✅ RSI_{self.period} 계산 완료!")
        print(f"   - 계산 기간: {self.period}일")
        print(f"   - 유효한 RSI 값: {df_copy[f'RSI_{self.period}'].notna().sum()}개")
        print(f"   - RSI 범위: {df_copy[f'RSI_{self.period}'].min():.2f} ~ {df_copy[f'RSI_{self.period}'].max():.2f}")
        
        return df_copy
    
    def analyze_rsi_signals(self, df, oversold=30, overbought=70):
        """
        RSI 기반 매매 신호를 분석합니다.
        
        Args:
            df (pd.DataFrame): RSI가 포함된 DataFrame
            oversold (float): 과매도 기준 (기본값: 30)
            overbought (float): 과매수 기준 (기본값: 70)
            
        Returns:
            pd.DataFrame: 매매 신호가 추가된 DataFrame
        """
        df_copy = df.copy()
        rsi_col = f'RSI_{self.period}'
        
        # 매매 신호 생성
        df_copy['RSI_Signal'] = 0  # 0: 보유, 1: 매수, -1: 매도
        
        # 과매도 구간에서 매수 신호
        df_copy.loc[df_copy[rsi_col] < oversold, 'RSI_Signal'] = 1
        
        # 과매수 구간에서 매도 신호  
        df_copy.loc[df_copy[rsi_col] > overbought, 'RSI_Signal'] = -1
        
        # 신호 통계
        buy_signals = (df_copy['RSI_Signal'] == 1).sum()
        sell_signals = (df_copy['RSI_Signal'] == -1).sum()
        
        print(f"📊 RSI 매매 신호 분석:")
        print(f"   - 과매도 기준: {oversold}")
        print(f"   - 과매수 기준: {overbought}")
        print(f"   - 매수 신호: {buy_signals}개")
        print(f"   - 매도 신호: {sell_signals}개")
        
        return df_copy
    
    def create_rsi_chart(self, df, symbol='BTCUSDT', save_html=True):
        """
        RSI 차트를 생성합니다.
        
        Args:
            df (pd.DataFrame): RSI와 신호가 포함된 DataFrame
            symbol (str): 심볼명
            save_html (bool): HTML 파일로 저장할지 여부
        """
        rsi_col = f'RSI_{self.period}'
        
        # 서브플롯 생성 (가격 차트 + RSI 차트)
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{symbol} 가격', f'RSI ({self.period})'),
            row_heights=[0.7, 0.3]
        )
        
        # 1. 가격 캔들스틱 차트
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='가격'
            ),
            row=1, col=1
        )
        
        # 2. RSI 라인 차트
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[rsi_col],
                mode='lines',
                name=f'RSI ({self.period})',
                line=dict(color='purple', width=2)
            ),
            row=2, col=1
        )
        
        # 3. RSI 기준선 (30, 70)
        fig.add_hline(y=30, line_dash="dash", line_color="green", 
                     annotation_text="과매도 (30)", row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", 
                     annotation_text="과매수 (70)", row=2, col=1)
        
        # 4. 매매 신호 표시
        if 'RSI_Signal' in df.columns:
            buy_points = df[df['RSI_Signal'] == 1]
            sell_points = df[df['RSI_Signal'] == -1]
            
            # 매수 신호
            if not buy_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=buy_points.index,
                        y=buy_points['close'],
                        mode='markers',
                        marker=dict(color='green', size=10, symbol='triangle-up'),
                        name='매수 신호'
                    ),
                    row=1, col=1
                )
            
            # 매도 신호
            if not sell_points.empty:
                fig.add_trace(
                    go.Scatter(
                        x=sell_points.index,
                        y=sell_points['close'],
                        mode='markers',
                        marker=dict(color='red', size=10, symbol='triangle-down'),
                        name='매도 신호'
                    ),
                    row=1, col=1
                )
        
        # 차트 레이아웃 설정
        fig.update_layout(
            title=f'{symbol} RSI 분석 차트',
            xaxis_title='시간',
            yaxis_title='가격 (USDT)',
            yaxis2_title='RSI',
            height=800,
            showlegend=True
        )
        
        # RSI 차트 Y축 범위 설정
        fig.update_yaxes(range=[0, 100], row=2, col=1)
        
        if save_html:
            filename = f'rsi_chart_{symbol.lower()}.html'
            fig.write_html(filename)
            print(f"📊 RSI 차트가 '{filename}'에 저장되었습니다.")
        
        return fig

def test_rsi_indicator():
    """RSI 지표 테스트 함수"""
    print("🚀 RSI 지표 계산 테스트 시작!")
    print("=" * 60)
    
    # 1. 데이터 수집
    collector = BitgetDataCollector()
    df = collector.fetch_historical_data('BTCUSDT', '1h', 30)
    
    if df is None:
        print("❌ 데이터 수집 실패!")
        return None
    
    print(f"📊 수집된 데이터: {len(df)}개 봉")
    
    # 2. RSI 지표 생성
    rsi_indicator = RSIIndicator(period=14)
    
    # 3. RSI 계산
    df_with_rsi = rsi_indicator.calculate_rsi(df)
    
    # 4. 매매 신호 분석
    df_with_signals = rsi_indicator.analyze_rsi_signals(df_with_rsi, oversold=30, overbought=70)
    
    # 5. 결과 미리보기
    print("\n📈 RSI 계산 결과 미리보기:")
    print(df_with_signals[['close', f'RSI_14', 'RSI_Signal']].tail(10))
    
    # 6. 차트 생성
    print("\n📊 RSI 차트 생성 중...")
    fig = rsi_indicator.create_rsi_chart(df_with_signals)
    
    # 7. 통계 요약
    print("\n📋 RSI 분석 요약:")
    print("=" * 40)
    rsi_col = f'RSI_14'
    print(f"평균 RSI: {df_with_signals[rsi_col].mean():.2f}")
    print(f"최고 RSI: {df_with_signals[rsi_col].max():.2f}")
    print(f"최저 RSI: {df_with_signals[rsi_col].min():.2f}")
    
    oversold_count = (df_with_signals[rsi_col] < 30).sum()
    overbought_count = (df_with_signals[rsi_col] > 70).sum()
    
    print(f"과매도 구간 (RSI < 30): {oversold_count}회 ({oversold_count/len(df_with_signals)*100:.1f}%)")
    print(f"과매수 구간 (RSI > 70): {overbought_count}회 ({overbought_count/len(df_with_signals)*100:.1f}%)")
    
    return df_with_signals

if __name__ == "__main__":
    result = test_rsi_indicator()
    
    if result is not None:
        print("\n✅ RSI 지표 테스트 완료!")
        print("📄 결과 파일:")
        print("   - rsi_chart_btcusdt.html (차트)")
    else:
        print("\n❌ RSI 지표 테스트 실패!")