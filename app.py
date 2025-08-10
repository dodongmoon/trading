from flask import Flask, request
import os
from dotenv import load_dotenv
import json

# main.py에서 주문 실행 함수와 주문 금액을 가져옵니다.
from main import execute_order, ORDER_AMOUNT_USDT

# .env 파일에서 환경 변수 로드
load_dotenv()

app = Flask(__name__)

# 외부에서 접속할 때 사용할 비밀 키 (간단한 보안 장치)
# 실제 운영 시에는 더 복잡하고 안전한 키를 사용해야 합니다.
SECRET_KEY = os.getenv("WEBHOOK_SECRET_KEY", "supersecretkey")

@app.route('/webhook', methods=['POST'])
def webhook():
    """
    트레이딩뷰에서 보낸 웹훅 요청을 처리하고, 신호에 따라 주문을 실행합니다.
    """
    data_str = request.data.decode('utf-8')
    print("-" * 50)
    print(f"웹훅 신호 수신: {data_str}")
    
    try:
        # 1. 수신된 데이터를 JSON으로 파싱
        data = json.loads(data_str)
        signal = data.get('signal', '').lower()  # 대소문자 구분 없이 처리
        strategy = data.get('strategy', 'default')  # 전략 이름
        symbol = data.get('ticker', 'BTCUSDT')  # 심볼 (확장성을 위해)
        
        print(f"📊 전략: {strategy}, 심볼: {symbol}, 신호: {signal}")
        
        if signal in ['buy', 'long', 'enter_long']:
            print(f"✅ 매수 신호 감지! {ORDER_AMOUNT_USDT} USDT 만큼 시장가 매수 주문을 실행합니다.")
            execute_order('buy', ORDER_AMOUNT_USDT)
        
        elif signal in ['sell', 'short', 'exit_long', 'close_long']:
            print(f"✅ 매도 신호 감지! 보유 포지션을 전량 청산합니다.")
            # 현재 포지션 확인
            from main import get_current_position
            position = get_current_position()
            if position and float(position.get('contracts', 0)) > 0:
                # 보유 수량만큼 매도 (포지션 청산)
                amount_to_sell = float(position['contracts'])
                current_price = float(position['markPrice'])
                usdt_amount = amount_to_sell * current_price
                print(f"포지션 청산: {amount_to_sell} BTC (≈{usdt_amount:.2f} USDT)")
                execute_order('sell', usdt_amount)
            else:
                print("⚠️ 청산할 포지션이 없습니다.")
        
        else:
            print(f"⚠️ 알 수 없는 신호({signal})입니다. 주문을 실행하지 않습니다.")

    except json.JSONDecodeError:
        print("🚨 오류: 수신된 데이터가 유효한 JSON 형식이 아닙니다.")
    except Exception as e:
        print(f"🚨 주문 처리 중 심각한 오류 발생: {e}")

    return "Webhook received successfully", 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002) 