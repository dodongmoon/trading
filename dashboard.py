import streamlit as st
import plotly.graph_objects as go
from main import get_bot_status, RSI_PERIOD  # main.py에서 함수와 변수 가져오기
import time

# --- 페이지 설정 ---
st.set_page_config(
    page_title="RSI 자동매매 봇 대시보드",
    page_icon="🤖",
    layout="wide",
)

# --- 제목 ---
st.title("🤖 RSI 자동매매 봇 대시보드")
st.caption("이 대시보드는 15초마다 자동으로 새로고침됩니다.")

# --- 플레이스홀더 생성 ---
# 나중에 이 위치에 실시간 데이터를 업데이트합니다.
placeholder = st.empty()


# --- 메인 대시보드 루프 ---
while True:
    status = get_bot_status()

    if status:
        with placeholder.container():
            # --- 1. 주요 지표 표시 ---
            col1, col2, col3 = st.columns(3)
            col1.metric(
                label=f"{status['symbol']} 현재가",
                value=f"${status['current_price']:,.2f}",
            )
            col2.metric(
                label=f"현재 RSI ({RSI_PERIOD})",
                value=f"{status['current_rsi']:.2f}",
                delta_color="off",
            )
            
            position_value = 0
            position_pnl = 0
            position_size = 0
            if status['position']:
                position_value = float(status['position']['info'].get('positionValue', 0))
                position_pnl = float(status['position']['info'].get('unrealisedPnl', 0))
                position_size = float(status['position']['contracts'])
                
            col3.metric(
                label="현재 포지션 가치 (PNL)",
                value=f"${position_value:,.2f}",
                delta=f"{position_pnl:,.2f} USDT",
            )

            st.divider()

            # --- 2. 차트 그리기 ---
            df = status["dataframe"]
            fig = go.Figure()

            # 가격 캔들스틱 차트
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="가격",
                )
            )
            
            # RSI 차트 (보조 y축)
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[f"RSI_{RSI_PERIOD}"],
                    name="RSI",
                    yaxis="y2",
                )
            )

            fig.update_layout(
                title=f"{status['symbol']} 실시간 차트",
                yaxis_title="가격 (USDT)",
                yaxis2=dict(
                    title="RSI",
                    overlaying="y",
                    side="right",
                    range=[0, 100],
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.divider()

            # --- 3. 포지션 상세 정보 표시 ---
            st.subheader("- 현재 보유 포지션 -")
            if status['position']:
                st.dataframe(status['position'], use_container_width=True)
            else:
                st.info("현재 보유한 포지션이 없습니다.")

    time.sleep(15)  # 15초 대기 