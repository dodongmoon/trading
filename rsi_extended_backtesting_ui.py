import streamlit as st
import pandas as pd
import numpy as np
from indicators_rsi import RSIIndicator
from extended_data_collector import ExtendedBitgetDataCollector
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class RSIExtendedBacktester:
    def __init__(self, initial_capital=10000):
        """
        확장된 RSI 백테스팅 엔진 초기화
        
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
        
    def run_backtest(self, df, rsi_period=14, oversold=30, overbought=70, trade_amount_ratio=1.0):
        """
        확장된 RSI 백테스팅 실행
        
        Args:
            df (pd.DataFrame): OHLCV 데이터
            rsi_period (int): RSI 계산 기간
            oversold (float): 과매도 기준
            overbought (float): 과매수 기준  
            trade_amount_ratio (float): 거래 시 사용할 자본 비율 (0.0~1.0)
            
        Returns:
            dict: 백테스팅 결과
        """
        self.reset()
        
        # RSI 계산
        rsi_indicator = RSIIndicator(period=rsi_period)
        df_with_rsi = rsi_indicator.calculate_rsi(df)
        
        rsi_col = f'RSI_{rsi_period}'
        
        for i in range(len(df_with_rsi)):
            current_price = df_with_rsi.iloc[i]['close']
            current_rsi = df_with_rsi.iloc[i][rsi_col]
            current_time = df_with_rsi.index[i]
            
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
            final_price = df_with_rsi.iloc[-1]['close']
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
                'timestamp': df_with_rsi.index[-1],
                'type': 'SELL (Final)',
                'price': final_price,
                'quantity': self.position,
                'amount': final_amount,
                'rsi': df_with_rsi.iloc[-1][rsi_col],
                'capital_after': self.capital,
                'portfolio_value': self.capital,
                'profit': profit,
                'profit_rate': profit_rate
            })
            
            self.position = 0
        
        return self.calculate_performance_metrics(df_with_rsi)
    
    def calculate_performance_metrics(self, df):
        """성과 지표 계산"""
        if not self.trades:
            # 거래가 없는 경우에도 기본 정보 반환
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,  # 변화 없음
                'total_return': 0,
                'total_return_rate': 0,
                'total_trades': 0,
                'win_rate': 0,
                'avg_profit_rate': 0,
                'max_profit': 0,
                'max_loss': 0,
                'backtest_days': (df.index[-1] - df.index[0]).days,
                'trades': [],
                'equity_curve': self.equity_curve,
                'data_points': len(df),
                'no_trades_reason': 'RSI 기준에 맞는 매매 신호가 발생하지 않았습니다.'
            }
            
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
        
        # 백테스팅 기간
        backtest_days = (df.index[-1] - df.index[0]).days
        
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
            'backtest_days': backtest_days,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'data_points': len(df)
        }

def create_extended_backtesting_ui():
    """확장된 RSI 백테스팅 Streamlit UI (3회 분할 지원)"""
    st.title("🚀 RSI 확장 백테스팅 시스템 (3회 분할 지원)")
    st.markdown("**API 제한을 우회하여 긴 기간 백테스팅이 가능합니다! (최대 3회 분할)**")
    st.markdown("---")
    
    # 사이드바 설정
    st.sidebar.header("⚙️ 백테스팅 설정")
    
    # 1. 기본 설정
    st.sidebar.subheader("📈 데이터 설정")
    symbol = st.sidebar.selectbox("거래 심볼", ["BTCUSDT"], index=0)
    timeframe = st.sidebar.selectbox("시간 간격", ["15m", "30m", "1h", "4h", "1d"], index=4)  # 기본값을 1d로
    
    # 3회 분할 기준 최대 기간 설정
    max_days_map = {
        '15m': 29,   # 15분봉: 최대 29일 (2850개 봉)
        '30m': 59,   # 30분봉: 최대 59일 (2850개 봉)
        '1h': 118,   # 1시간봉: 최대 118일 (2850개 봉)
        '4h': 256,   # 4시간봉: 최대 256일 (1539개 봉)
        '1d': 255    # 1일봉: 최대 255일 (255개 봉)
    }
    
    # 봉수 정보 맵
    candle_info_map = {
        '15m': "2,850개",
        '30m': "2,850개", 
        '1h': "2,850개",
        '4h': "1,539개",
        '1d': "255개"
    }
    
    max_days = max_days_map[timeframe]
    candle_info = candle_info_map[timeframe]
    
    # 안내 메시지 표시
    st.sidebar.info(f"💡 **{timeframe}봉**은 최대 **{candle_info}**의 봉까지만 호출이 가능해요!")
    
    days = st.sidebar.slider(f"백테스팅 기간 (일) - 최대 {max_days}일", 
                           min_value=7, 
                           max_value=max_days, 
                           value=min(90, max_days))
    
    # 2. 자본 설정
    st.sidebar.subheader("💰 자본 설정")
    initial_capital = st.sidebar.number_input("초기 자본금 (USDT)", 1000, 100000, 10000, 1000)
    trade_ratio = st.sidebar.slider("거래 시 사용 자본 비율 (%)", 10, 100, 100, 5) / 100
    
    # 3. RSI 설정
    st.sidebar.subheader("🔢 RSI 설정")
    rsi_period = st.sidebar.slider("RSI 기간", 5, 30, 14)
    oversold = st.sidebar.slider("과매도 기준", 10, 40, 30)
    overbought = st.sidebar.slider("과매수 기준", 60, 90, 70)
    
    # 예상 분할 횟수 표시 (3회 분할 기준)
    if timeframe in ['15m', '30m', '1h']:
        # 15분, 30분, 1시간봉: 950개씩 최대 3회
        if timeframe == '15m':
            days_per_call = 950 // 96  # 15분봉: 하루 96개
        elif timeframe == '30m':
            days_per_call = 950 // 48  # 30분봉: 하루 48개
        else:  # 1h
            days_per_call = 950 // 24  # 1시간봉: 하루 24개
    elif timeframe == '4h':
        # 4시간봉: 513개씩 최대 3회
        days_per_call = 513 // 6  # 4시간봉: 하루 6개
    else:  # 1d
        # 1일봉: 85개씩 최대 3회
        days_per_call = 85  # 1일봉: 하루 1개
    
    expected_splits = min(3, max(1, (days + days_per_call - 1) // days_per_call))
    
    if expected_splits >= 3:
        st.sidebar.warning(f"📊 예상 API 요청: {expected_splits}회 (최대 분할) - 시간이 오래 걸릴 수 있습니다")
    elif expected_splits > 1:
        st.sidebar.info(f"📊 예상 API 요청: {expected_splits}회 (분할 수집)")
    else:
        st.sidebar.success(f"📊 예상 API 요청: {expected_splits}회 (단일 수집)")
    
    # 백테스팅 실행 버튼
    if st.sidebar.button("🚀 확장 백테스팅 시작", type="primary"):
        
        with st.spinner("📡 확장 데이터 수집 중... (시간이 오래 걸릴 수 있습니다)"):
            # 확장 데이터 수집
            collector = ExtendedBitgetDataCollector()
            df = collector.fetch_historical_data_extended(symbol, timeframe, days)
            
            if df is None:
                st.error("❌ 데이터 수집에 실패했습니다.")
                return
            
            st.success(f"✅ 데이터 수집 완료! {len(df)}개 봉, {(df.index[-1] - df.index[0]).days}일")
        
        with st.spinner("🧮 RSI 계산 및 백테스팅 실행 중..."):
            # 백테스팅 실행
            backtester = RSIExtendedBacktester(initial_capital)
            results = backtester.run_backtest(df, rsi_period, oversold, overbought, trade_ratio)
            
            if results is None:
                st.error("❌ 백테스팅 실행에 실패했습니다.")
                return
        
        # 결과 표시
        st.success("✅ 확장 백테스팅 완료!")
        
        # 거래가 없는 경우 특별 처리
        if results.get('total_trades', 0) == 0:
            st.warning("⚠️ 설정한 RSI 기준에 맞는 매매 신호가 발생하지 않았습니다!")
            st.info(f"💡 **해결 방법**: RSI 기준을 조정해보세요 (과매도: {oversold-5}~{oversold+5}, 과매수: {overbought-5}~{overbought+5})")
            
            # 현재 RSI 분포 정보 표시
            if results.get('equity_curve'):
                equity_df = pd.DataFrame(results['equity_curve'])
                rsi_values = equity_df['rsi'].dropna()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("RSI 최소값", f"{rsi_values.min():.1f}")
                with col2:
                    st.metric("RSI 최대값", f"{rsi_values.max():.1f}")
                with col3:
                    st.metric("RSI 평균값", f"{rsi_values.mean():.1f}")
                
                st.write(f"- **과매도 신호** (RSI < {oversold}): {(rsi_values < oversold).sum()}회")
                st.write(f"- **과매수 신호** (RSI > {overbought}): {(rsi_values > overbought).sum()}회")
        
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
                "📅 백테스팅 기간",
                f"{results['backtest_days']}일",
                f"{results['data_points']}개 봉"
            )
        
        # 2. 상세 분석
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 수익/손실 분석")
            st.write(f"**최종 자본:** {results['final_capital']:,.0f} USDT")
            st.write(f"**총 수익률:** {results['total_return_rate']:+.2f}%")
            st.write(f"**최대 수익:** {results['max_profit']:+,.0f} USDT")
            st.write(f"**최대 손실:** {results['max_loss']:+,.0f} USDT")
            
            # 연간 수익률 계산
            if results['backtest_days'] > 0:
                annual_return = (results['total_return_rate'] * 365) / results['backtest_days']
                st.write(f"**연간 수익률 (추정):** {annual_return:+.1f}%")
            
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
            st.write(f"**기간:** {results['backtest_days']}일 ({timeframe})")
            st.write(f"**RSI 기간:** {rsi_period}일")
            st.write(f"**과매도/과매수:** {oversold}/{overbought}")
            st.write(f"**거래 자본 비율:** {trade_ratio*100:.0f}%")
            st.write(f"**데이터 포인트:** {results['data_points']}개")
            
            # 성과 지표
            st.subheader("📈 성과 지표")
            if results['backtest_days'] > 0:
                trades_per_month = (results['total_trades'] * 30) / results['backtest_days']
                st.write(f"**월평균 거래 횟수:** {trades_per_month:.1f}회")
                
                if results['total_trades'] > 0:
                    avg_holding_days = results['backtest_days'] / (results['total_trades'] * 2)
                    st.write(f"**평균 보유 기간:** {avg_holding_days:.1f}일")
        
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
                title=f"{symbol} 확장 RSI 백테스팅 결과 ({results['backtest_days']}일)",
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
        
        # 5. 다운로드 옵션
        st.markdown("---")
        st.subheader("💾 결과 다운로드")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 거래 내역 CSV 다운로드"):
                if results['trades']:
                    trades_df = pd.DataFrame(results['trades'])
                    csv = trades_df.to_csv(index=False)
                    st.download_button(
                        label="다운로드",
                        data=csv,
                        file_name=f"rsi_trades_{symbol}_{timeframe}_{days}d.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("📈 자본 변화 CSV 다운로드"):
                if results['equity_curve']:
                    equity_df = pd.DataFrame(results['equity_curve'])
                    csv = equity_df.to_csv(index=False)
                    st.download_button(
                        label="다운로드",
                        data=csv,
                        file_name=f"rsi_equity_{symbol}_{timeframe}_{days}d.csv",
                        mime="text/csv"
                    )

if __name__ == "__main__":
    create_extended_backtesting_ui()