# RSI (Relative Strength Index) 계산 원리 설명

def explain_rsi_calculation():
    """
    RSI 계산에 대한 정확한 설명
    
    RSI는 오직 '가격 변화'만을 사용합니다!
    거래량은 전혀 사용하지 않습니다.
    """
    
    print("📊 RSI 계산 공식:")
    print("=" * 50)
    print("1️⃣ 상승폭 계산: 전일 대비 상승한 경우의 변화량")
    print("2️⃣ 하락폭 계산: 전일 대비 하락한 경우의 변화량") 
    print("3️⃣ 평균 상승폭 = 14일간 상승폭의 평균")
    print("4️⃣ 평균 하락폭 = 14일간 하락폭의 평균")
    print("5️⃣ RS = 평균 상승폭 / 평균 하락폭")
    print("6️⃣ RSI = 100 - (100 / (1 + RS))")
    print("\n✅ 보시다시피 거래량(Volume)은 전혀 사용되지 않습니다!")
    
    print("\n🔍 거래량을 사용하는 지표들:")
    print("- OBV (On-Balance Volume)")
    print("- VWAP (Volume Weighted Average Price)")
    print("- Money Flow Index (MFI)")
    print("- Accumulation/Distribution Line")
    
    print("\n🚫 거래량을 사용하지 않는 지표들:")
    print("- RSI ✅")
    print("- MACD ✅") 
    print("- 볼린저 밴드 ✅")
    print("- 이동평균선 ✅")
    print("- 스토캐스틱 ✅")

# 실제 RSI 계산 예시
import pandas as pd
import pandas_ta as ta

def calculate_rsi_example(df):
    """실제 RSI 계산 예시"""
    print("\n🧮 실제 RSI 계산 예시:")
    print("=" * 50)
    
    # RSI 계산 (오직 종가만 사용)
    df['RSI'] = ta.rsi(df['close'], length=14)
    
    print("📈 사용된 데이터:")
    print("- close (종가): ✅ 사용됨")
    print("- volume (거래량): ❌ 사용 안됨")
    print("- open, high, low: ❌ 사용 안됨")
    
    print("\n📊 RSI 결과 샘플:")
    print(df[['close', 'volume', 'RSI']].tail())
    
    return df

if __name__ == "__main__":
    explain_rsi_calculation()
    
    # 실제 데이터로 테스트
    print("\n" + "=" * 60)
    print("🧪 실제 데이터로 RSI 계산 테스트")
    
    # CSV 파일 읽기
    df = pd.read_csv('btc_1h_30days.csv', index_col='timestamp', parse_dates=True)
    df = calculate_rsi_example(df)