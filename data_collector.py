import ccxt
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

# .env 파일에서 API 키 로드
load_dotenv()

class BitgetDataCollector:
    def __init__(self):
        """Bitget 데이터 수집기 초기화"""
        self.exchange = ccxt.bitget({
            'apiKey': os.getenv("BITGET_API_KEY"),
            'secret': os.getenv("BITGET_API_SECRET"),
            'password': os.getenv("BITGET_PASSPHRASE"),
            'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True,
            },
        })
        # 모의 투자 서버 설정
        self.exchange.set_sandbox_mode(True)
        
    def fetch_historical_data(self, symbol='BTCUSDT', timeframe='1h', days=30):
        """
        역사적 OHLCV 데이터를 수집합니다.
        
        Args:
            symbol (str): 거래 심볼 (예: 'BTCUSDT')
            timeframe (str): 시간 간격 ('1m', '5m', '15m', '1h', '4h', '1d')
            days (int): 수집할 일수
            
        Returns:
            pd.DataFrame: OHLCV 데이터가 포함된 DataFrame
        """
        try:
            print(f"📊 {symbol} {timeframe} 데이터 수집 중... (최근 {days}일)")
            
            # 시작 시간 계산 (밀리초 단위)
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            # 한번에 가져올 수 있는 최대 캔들 수 (보통 1000개)
            limit = 1000
            all_ohlcv = []
            
            while True:
                # OHLCV 데이터 요청
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                # 마지막 캔들의 시간을 다음 요청의 시작점으로 설정
                since = ohlcv[-1][0] + 1
                
                # 현재 시간을 넘어서면 중단
                if since > int(datetime.now().timestamp() * 1000):
                    break
                    
                # API 호출 제한을 위한 대기
                time.sleep(0.1)
                
                print(f"수집된 캔들 수: {len(all_ohlcv)}")
            
            # DataFrame으로 변환
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # 중복 제거 및 정렬
            df = df.drop_duplicates().sort_index()
            
            print(f"✅ 데이터 수집 완료!")
            print(f"   - 심볼: {symbol}")
            print(f"   - 기간: {df.index[0]} ~ {df.index[-1]}")
            print(f"   - 총 캔들 수: {len(df)}")
            print(f"   - 시간 간격: {timeframe}")
            
            return df
            
        except Exception as e:
            print(f"❌ 데이터 수집 중 오류 발생: {e}")
            return None
    
    def get_multiple_timeframes(self, symbol='BTCUSDT', timeframes=['1h', '4h', '1d'], days=30):
        """
        여러 시간 간격의 데이터를 동시에 수집합니다.
        
        Args:
            symbol (str): 거래 심볼
            timeframes (list): 시간 간격 리스트
            days (int): 수집할 일수
            
        Returns:
            dict: 시간 간격별 DataFrame 딕셔너리
        """
        data_dict = {}
        
        for timeframe in timeframes:
            print(f"\n🔄 {timeframe} 데이터 수집 중...")
            df = self.fetch_historical_data(symbol, timeframe, days)
            if df is not None:
                data_dict[timeframe] = df
            time.sleep(1)  # API 호출 간격 조절
            
        return data_dict
    
    def save_data_to_csv(self, df, filename):
        """데이터를 CSV 파일로 저장합니다."""
        try:
            df.to_csv(filename)
            print(f"💾 데이터가 {filename}에 저장되었습니다.")
        except Exception as e:
            print(f"❌ 데이터 저장 중 오류 발생: {e}")

# 테스트 함수
def test_data_collection():
    """데이터 수집 테스트"""
    collector = BitgetDataCollector()
    
    # 1시간 봉 30일치 데이터 수집
    df = collector.fetch_historical_data('BTCUSDT', '1h', 30)
    
    if df is not None:
        print("\n📈 수집된 데이터 미리보기:")
        print(df.head())
        print(f"\n📊 데이터 통계:")
        print(df.describe())
        
        # CSV로 저장
        collector.save_data_to_csv(df, 'btc_1h_30days.csv')
        
        return df
    else:
        print("❌ 데이터 수집에 실패했습니다.")
        return None

if __name__ == "__main__":
    test_data_collection()