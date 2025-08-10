import ccxt
import os
from dotenv import load_dotenv

# .env 파일에서 API 키, 시크릿, 패스프레이즈 로드
load_dotenv()

API_KEY = os.getenv("BITGET_API_KEY")
API_SECRET = os.getenv("BITGET_API_SECRET")
API_PASSPHRASE = os.getenv("BITGET_PASSPHRASE")

# --- 설정 ---
SYMBOL = "BTCUSDT"  # API 문서에 맞게 슬래시 제거
ORDER_AMOUNT_USDT = 100  # 한번에 주문할 금액 (USDT) - 최소 주문 수량 충족을 위해 100로 상향

# --- 거래소 객체 초기화 (포지션 조회용) ---
exchange = ccxt.bitget({
    'apiKey': API_KEY,
    'secret': API_SECRET,
    'password': API_PASSPHRASE,
    'options': {
        'defaultType': 'swap',
        'adjustForTimeDifference': True, # 시간 동기화 문제 해결
    },
})
# 모의 투자 서버 설정
exchange.set_sandbox_mode(True)

def get_current_position():
    """현재 보유 중인 포지션 정보를 반환합니다."""
    try:
        positions = exchange.fetch_positions([SYMBOL])
        btc_position = next((p for p in positions if p['info']['symbol'] == SYMBOL), None)
        return btc_position
    except Exception as e:
        print(f"포지션 정보 조회 중 오류 발생: {e}")
        return None

def execute_order(side, amount_usdt):
    try:
        # 1. 현재 가격 조회
        ticker = exchange.fetch_ticker(SYMBOL)
        current_price = ticker['last']
        if not current_price:
            print("현재 가격을 가져올 수 없습니다.")
            return None

        # 2. 주문 수량 계산 (USDT -> BTC)
        amount = amount_usdt / current_price

        print(f"주문 실행: {side.upper()} {amount:.6f} BTC (≈{amount_usdt} USDT) at market price...")
        
        # 3. ccxt를 통한 주문 실행 (sandbox 모드 자동 적용)
        # Bitget 단방향 포지션 모드를 위한 추가 파라미터
        params = {
            'tradeSide': 'open',  # 포지션 열기
            'marginCoin': 'USDT',  # 마진 코인
            'marginMode': 'crossed'  # 교차 마진
        }
        
        if side == 'buy':
            order = exchange.create_market_buy_order(SYMBOL, amount, params=params)
        else:
            order = exchange.create_market_sell_order(SYMBOL, amount, params=params)
        
        print("✅ 주문 성공!")
        print(f"주문 정보: {order}")
        return order

    except ccxt.InsufficientFunds as e:
        print(f"❌ 오류: 잔고가 부족합니다. {e}")
    except ccxt.NetworkError as e:
        print(f"❌ 네트워크 오류: {e}")
    except ccxt.ExchangeError as e:
        print(f"❌ 거래소 오류: {e}")
    except Exception as e:
        print(f"❌ 주문 실행 중 오류 발생: {e}")
    
    return None

def check_recent_orders():
    """최근 주문 내역을 확인합니다."""
    try:
        orders = exchange.fetch_orders(SYMBOL, limit=5)
        print("=== 최근 주문 내역 ===")
        for order in orders:
            print(f"주문ID: {order['id']}, 상태: {order['status']}, 타입: {order['side']}, 수량: {order['amount']}")
        return orders
    except Exception as e:
        print(f"주문 내역 조회 오류: {e}")
        return None

if __name__ == '__main__':
    # 이 파일은 직접 실행되지 않고, 다른 파일(app.py)에서 함수를 임포트하여 사용됩니다.
    # 테스트를 위해 아래 함수들을 호출해볼 수 있습니다.
    
    print("현재 포지션 정보:")
    position = get_current_position()
    if position:
        print(position)
    else:
        print("조회 실패 또는 포지션 없음")

    # # 주문 테스트 (실제 주문이 체결되므로 주의!)
    # print("\n매수 주문 테스트:")
    # execute_order('buy', ORDER_AMOUNT_USDT)
    
    # # 잠시 대기 후 포지션 재확인
    # import time
    # time.sleep(5) 
    
    # print("\n매도 주문 테스트:")
    # execute_order('sell', ORDER_AMOUNT_USDT) 

    # 테스트: 최근 주문 확인
    check_recent_orders() 