import streamlit as st
import pandas as pd
import numpy as np
from indicators_rsi import RSIIndicator
from data_collector import BitgetDataCollector
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class RSIBacktester:
    def __init__(self, initial_capital=10000):
        """
        RSI 백테스팅 엔진 초기화
        
        Args:
            initial_capital (float): 초기 자본금
        """
        self.initial_capital = initial_capital
        self.reset()
        
    def reset(self):
        """백테스팅 상태 초기화"""
        self.capital = self.initial_capital
        self.position = 0  # 보유 수량
        self.position_value = 0  # 포지션 가치
        self.trades = []  # 거래 내역
        self.equity_curve = []  # 자본 변화 곡선
        
    def run_backtest(self, df, oversold=30, overbought=70, trade_amount_ratio=1.0):
        """
        RSI 백테스팅 실행
        
        Args:
            df (pd.DataFrame): RSI 신호가 포함된 데이터
            oversold (float): 과매도 기준
            overbought (float): 과매수 기준  
            trade_amount_ratio (float): 거래 시 사용할 자본 비율 (0.0~1.0)
            
        Returns:
            dict: 백테스팅 결과
        """
        self.reset()
        
        rsi_col = 'RSI_14'
        
        for i in range(len(df)):
            current_price = df.iloc[i]['close']
            current_rsi = df.iloc[i][rsi_col]
            current_time = df.index[i]
            
            # 현재 포트폴리오 가치 계산
            portfolio_value = self.capital + (self.position * current_price)
            self.equity_curve.append({
                'timestamp': current_time,
                'portfolio_value': portfolio_value,
                'price': current_price,
                'rsi': current_rsi
            })
            
            # 매수 신호 (과매도 + 현재 포지션 없음)
            if current_rsi < oversold and self.position == 0:
                trade_amount = self.capital * trade_amount_ratio
                if trade_amount > 0:
                    shares_to_buy = trade_amount / current_price
                    self.position = shares_to_buy
                    self.capital -= trade_amount
                    
                    self.trades.append({
                        'timestamp': current_time,
                        'type': 'BUY',
                        'price': current_price,
                        'quantity': shares_to_buy,
                        'amount': trade_amount,
                        'rsi': current_rsi,
                        'capital_after': self.capital,
                        'portfolio_value': self.capital + (self.position * current_price)
                    })
            
            # 매도 신호 (과매수 + 현재 포지션 있음)
            elif current_rsi > overbought and self.position > 0:
                sell_amount = self.position * current_price
                self.capital += sell_amount
                
                # 수익률 계산 (마지막 매수 거래와 비교)
                last_buy = None
                for trade in reversed(self.trades):
                    if trade['type'] == 'BUY':
                        last_buy = trade
                        break
                
                profit = 0
                profit_rate = 0
                if last_buy:
                    profit = sell_amount - last_buy['amount']
                    profit_rate = (profit / last_buy['amount']) * 100
                
                self.trades.append({
                    'timestamp': current_time,
                    'type': 'SELL',
                    'price': current_price,
                    'quantity': self.position,
                    'amount': sell_amount,
                    'rsi': current_rsi,
                    'capital_after': self.capital,
                    'portfolio_value': self.capital,
                    'profit': profit,
                    'profit_rate': profit_rate
                })
                
                self.position = 0
        
        # 최종 포지션이 남아있으면 마지막 가격으로 청산
        if self.position > 0:
            final_price = df.iloc[-1]['close']
            final_amount = self.position * final_price
            self.capital += final_amount
            
            # 마지막 청산 거래 기록
            last_buy = None
            for trade in reversed(self.trades):
                if trade['type'] == 'BUY':
                    last_buy = trade
                    break
            
            profit = 0
            profit_rate = 0
            if last_buy:
                profit = final_amount - last_buy['amount']
                profit_rate = (profit / last_buy['amount']) * 100
            
            self.trades.append({
                'timestamp': df.index[-1],
                'type': 'SELL (Final)',
                'price': final_price,
                'quantity': self.position,
                'amount': final_amount,
                'rsi': df.iloc[-1][rsi_col],
                'capital_after': self.capital,
                'portfolio_value': self.capital,
                'profit': profit,
                'profit_rate': profit_rate
            })
            
            self.position = 0
        
        return self.calculate_performance_metrics()
    
    def calculate_performance_metrics(self):
        """성과 지표 계산"""
        if not self.trades:
            return None
            
        # 기본 지표
        total_return = self.capital - self.initial_capital
        total_return_rate = (total_return / self.initial_capital) * 100
        
        # 거래 분석
        buy_trades = [t for t in self.trades if t['type'] == 'BUY']
        sell_trades = [t for t in self.trades if t['type'] in ['SELL', 'SELL (Final)']]
        
        total_trades = len(buy_trades)
        
        # 승률 계산
        profitable_trades = [t for t in sell_trades if t.get('profit', 0) > 0]
        win_rate = (len(profitable_trades) / len(sell_trades) * 100) if sell_trades else 0
        
        # 평균 수익률
        avg_profit_rate = np.mean([t.get('profit_rate', 0) for t in sell_trades]) if sell_trades else 0
        
        # 최대 수익/손실
        max_profit = max([t.get('profit', 0) for t in sell_trades]) if sell_trades else 0
        max_loss = min([t.get('profit', 0) for t in sell_trades]) if sell_trades else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_capital': self.capital,
            'total_return': total_return,
            'total_return_rate': total_return_rate,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_profit_rate': avg_profit_rate,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'trades': self.trades,
            'equity_curve': self.equity_curve
        }

def create_backtesting_ui():
    """RSI 백테스팅 Streamlit UI"""
    st.title("📊 RSI 백테스팅 시스템")
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 백테스팅 설정")
    
    # 1. 기본 설정
    st.sidebar.subheader("📈 데이터 설정")
    symbol = st.sidebar.selectbox("거래 심볼", ["BTCUSDT"], index=0)
    timeframe = st.sidebar.selectbox("시간 간격", ["1h", "4h", "1d"], index=0)
    days = st.sidebar.slider("백테스팅 기간 (일)", 7, 90, 30)
    
    # 2. 자본 설정
    st.sidebar.subheader("💰 자본 설정")
    initial_capital = st.sidebar.number_input("초기 자본금 (USDT)", 1000, 100000, 10000, 1000)
    trade_ratio = st.sidebar.slider("거래 시 사용 자본 비율 (%)", 10, 100, 100, 5) / 100
    
    # 3. RSI 설정
    st.sidebar.subheader("🔢 RSI 설정")
    rsi_period = st.sidebar.slider("RSI 기간", 5, 30, 14)
    oversold = st.sidebar.slider("과매도 기준", 10, 40, 30)
    overbought = st.sidebar.slider("과매수 기준", 60, 90, 70)
    
    # 백테스팅 실행 버튼
    if st.sidebar.button("🚀 백테스팅 시작", type="primary"):
        
        with st.spinner("데이터 수집 중..."):
            # 데이터 수집
            collector = BitgetDataCollector()
            df = collector.fetch_historical_data(symbol, timeframe, days)
            
            if df is None:
                st.error("❌ 데이터 수집에 실패했습니다.")
                return
        
        with st.spinner("RSI 계산 중..."):
            # RSI 계산
            rsi_indicator = RSIIndicator(period=rsi_period)
            df_with_rsi = rsi_indicator.calculate_rsi(df)
        
        with st.spinner("백테스팅 실행 중..."):
            # 백테스팅 실행
            backtester = RSIBacktester(initial_capital)
            results = backtester.run_backtest(df_with_rsi, oversold, overbought, trade_ratio)
            
            if results is None:
                st.error("❌ 백테스팅 실행에 실패했습니다.")
                return
        
        # 결과 표시
        st.success("✅ 백테스팅 완료!")
        
        # 1. 성과 요약
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "💰 총 수익",
                f"{results['total_return']:,.0f} USDT",
                f"{results['total_return_rate']:+.2f}%"
            )
        
        with col2:
            st.metric(
                "📊 총 거래 횟수",
                f"{results['total_trades']}회",
                f"승률 {results['win_rate']:.1f}%"
            )
        
        with col3:
            st.metric(
                "📈 평균 수익률",
                f"{results['avg_profit_rate']:+.2f}%",
                f"거래당 평균"
            )
        
        with col4:
            st.metric(
                "💵 최종 자본",
                f"{results['final_capital']:,.0f} USDT",
                f"초기 {results['initial_capital']:,.0f} USDT"
            )
        
        # 2. 상세 분석
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 수익/손실 분석")
            st.write(f"**최대 수익:** {results['max_profit']:+,.0f} USDT")
            st.write(f"**최대 손실:** {results['max_loss']:+,.0f} USDT")
            st.write(f"**수익률:** {results['total_return_rate']:+.2f}%")
            
            # 거래별 수익률 분포
            if results['trades']:
                sell_trades = [t for t in results['trades'] if 'profit_rate' in t]
                if sell_trades:
                    profit_rates = [t['profit_rate'] for t in sell_trades]
                    
                    fig_hist = go.Figure(data=[go.Histogram(x=profit_rates, nbinsx=20)])
                    fig_hist.update_layout(
                        title="거래별 수익률 분포",
                        xaxis_title="수익률 (%)",
                        yaxis_title="거래 횟수",
                        height=300
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            st.subheader("⚙️ 백테스팅 설정")
            st.write(f"**심볼:** {symbol}")
            st.write(f"**기간:** {days}일 ({timeframe})")
            st.write(f"**RSI 기간:** {rsi_period}일")
            st.write(f"**과매도/과매수:** {oversold}/{overbought}")
            st.write(f"**거래 자본 비율:** {trade_ratio*100:.0f}%")
            st.write(f"**데이터 포인트:** {len(df)}개")
        
        # 3. 자본 변화 곡선
        st.markdown("---")
        st.subheader("📈 포트폴리오 가치 변화")
        
        if results['equity_curve']:
            equity_df = pd.DataFrame(results['equity_curve'])
            
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=("포트폴리오 가치", "RSI"),
                row_heights=[0.7, 0.3]
            )
            
            # 포트폴리오 가치
            fig.add_trace(
                go.Scatter(
                    x=equity_df['timestamp'],
                    y=equity_df['portfolio_value'],
                    mode='lines',
                    name='포트폴리오 가치',
                    line=dict(color='blue', width=2)
                ),
                row=1, col=1
            )
            
            # 초기 자본선
            fig.add_hline(
                y=initial_capital,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"초기 자본 ({initial_capital:,} USDT)",
                row=1, col=1
            )
            
            # RSI
            fig.add_trace(
                go.Scatter(
                    x=equity_df['timestamp'],
                    y=equity_df['rsi'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=1)
                ),
                row=2, col=1
            )
            
            # RSI 기준선
            fig.add_hline(y=oversold, line_dash="dash", line_color="green", row=2, col=1)
            fig.add_hline(y=overbought, line_dash="dash", line_color="red", row=2, col=1)
            
            # 거래 신호 표시
            buy_trades = [t for t in results['trades'] if t['type'] == 'BUY']
            sell_trades = [t for t in results['trades'] if t['type'] in ['SELL', 'SELL (Final)']]
            
            if buy_trades:
                buy_times = [t['timestamp'] for t in buy_trades]
                buy_values = [t['portfolio_value'] for t in buy_trades]
                
                fig.add_trace(
                    go.Scatter(
                        x=buy_times,
                        y=buy_values,
                        mode='markers',
                        marker=dict(color='green', size=8, symbol='triangle-up'),
                        name='매수'
                    ),
                    row=1, col=1
                )
            
            if sell_trades:
                sell_times = [t['timestamp'] for t in sell_trades]
                sell_values = [t['portfolio_value'] for t in sell_trades]
                
                fig.add_trace(
                    go.Scatter(
                        x=sell_times,
                        y=sell_values,
                        mode='markers',
                        marker=dict(color='red', size=8, symbol='triangle-down'),
                        name='매도'
                    ),
                    row=1, col=1
                )
            
            fig.update_layout(
                title=f"{symbol} RSI 백테스팅 결과",
                height=600,
                showlegend=True
            )
            
            fig.update_yaxes(range=[0, 100], row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 4. 거래 내역
        st.markdown("---")
        st.subheader("📋 거래 내역")
        
        if results['trades']:
            trades_df = pd.DataFrame(results['trades'])
            
            # 컬럼 정리
            display_columns = ['timestamp', 'type', 'price', 'quantity', 'amount', 'rsi']
            if 'profit' in trades_df.columns:
                display_columns.extend(['profit', 'profit_rate'])
            
            trades_display = trades_df[display_columns].copy()
            trades_display['timestamp'] = trades_display['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            trades_display['price'] = trades_display['price'].round(2)
            trades_display['quantity'] = trades_display['quantity'].round(6)
            trades_display['amount'] = trades_display['amount'].round(2)
            trades_display['rsi'] = trades_display['rsi'].round(2)
            
            if 'profit' in trades_display.columns:
                trades_display['profit'] = trades_display['profit'].round(2)
                trades_display['profit_rate'] = trades_display['profit_rate'].round(2)
            
            st.dataframe(trades_display, use_container_width=True)
        else:
            st.info("거래 내역이 없습니다.")

if __name__ == "__main__":
    create_backtesting_ui()