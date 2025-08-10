import ccxt
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time

# .env 파일에서 API 키 로드
load_dotenv()

class BitgetAPILimitsTester:
    def __init__(self):
        """Bitget API 제한 테스터 초기화"""
        self.exchange = ccxt.bitget({
            'apiKey': os.getenv("BITGET_API_KEY"),
            'secret': os.getenv("BITGET_API_SECRET"), 
            'password': os.getenv("BITGET_PASSPHRASE"),
            'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True,
            },
        })
        self.exchange.set_sandbox_mode(True)
        
    def test_max_candles_per_request(self, symbol='BTCUSDT', timeframe='1h'):
        """한 번에 가져올 수 있는 최대 봉 수 테스트"""
        print(f"🧪 {timeframe} 봉 최대 개수 테스트 중...")
        
        test_limits = [500, 1000, 1500, 2000, 5000]
        max_working_limit = 0
        
        for limit in test_limits:
            try:
                print(f"   📊 {limit}개 봉 요청 중...")
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                actual_count = len(ohlcv)
                
                print(f"   ✅ 요청: {limit}개, 실제 수신: {actual_count}개")
                
                if actual_count > 0:
                    max_working_limit = max(max_working_limit, actual_count)
                    
                time.sleep(1)  # API 제한 방지
                
            except Exception as e:
                print(f"   ❌ {limit}개 요청 실패: {e}")
                break
                
        print(f"🎯 {timeframe} 봉 최대 개수: {max_working_limit}개")
        return max_working_limit
    
    def test_all_timeframes_limits(self, symbol='BTCUSDT'):
        """모든 시간 간격별 제한 테스트"""
        timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']
        limits_info = {}
        
        print("🔍 모든 시간 간격별 제한 테스트 시작...")
        print("=" * 60)
        
        for timeframe in timeframes:
            max_candles = self.test_max_candles_per_request(symbol, timeframe)
            
            # 시간 간격별 실제 커버 가능 기간 계산
            if timeframe == '1m':
                max_period_hours = max_candles / 60
                max_period_days = max_period_hours / 24
            elif timeframe == '5m':
                max_period_hours = max_candles * 5 / 60
                max_period_days = max_period_hours / 24
            elif timeframe == '15m':
                max_period_hours = max_candles * 15 / 60
                max_period_days = max_period_hours / 24
            elif timeframe == '1h':
                max_period_hours = max_candles
                max_period_days = max_period_hours / 24
            elif timeframe == '4h':
                max_period_hours = max_candles * 4
                max_period_days = max_period_hours / 24
            elif timeframe == '1d':
                max_period_days = max_candles
                max_period_hours = max_period_days * 24
            
            limits_info[timeframe] = {
                'max_candles': max_candles,
                'max_hours': round(max_period_hours, 2),
                'max_days': round(max_period_days, 2)
            }
            
            print(f"📅 {timeframe}: {max_candles}개 봉 = {max_period_days:.1f}일 = {max_period_hours:.1f}시간")
            print("-" * 40)
            
            time.sleep(2)  # API 제한 방지
            
        return limits_info
    
    def test_api_rate_limits(self, symbol='BTCUSDT', timeframe='1h'):
        """API 호출 속도 제한 테스트"""
        print("⚡ API 호출 속도 제한 테스트 중...")
        
        request_times = []
        max_requests = 10
        
        start_time = time.time()
        
        for i in range(max_requests):
            request_start = time.time()
            
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
                request_end = time.time()
                
                request_duration = request_end - request_start
                request_times.append(request_duration)
                
                print(f"   요청 {i+1}: {request_duration:.3f}초")
                
                # 너무 빠르면 대기
                if request_duration < 0.1:
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"   ❌ 요청 {i+1} 실패: {e}")
                break
        
        total_time = time.time() - start_time
        avg_request_time = sum(request_times) / len(request_times) if request_times else 0
        
        print(f"📊 총 {len(request_times)}개 요청 완료")
        print(f"📊 총 소요 시간: {total_time:.2f}초")
        print(f"📊 평균 요청 시간: {avg_request_time:.3f}초")
        print(f"📊 초당 요청 수: {len(request_times) / total_time:.2f} req/sec")
        
        return {
            'total_requests': len(request_times),
            'total_time': total_time,
            'avg_request_time': avg_request_time,
            'requests_per_second': len(request_times) / total_time
        }
    
    def generate_ui_constraints_config(self, limits_info):
        """UI에서 사용할 제약 조건 설정 생성"""
        print("\n🎨 UI 제약 조건 설정 생성 중...")
        
        config = {
            'timeframe_limits': {},
            'recommended_periods': {}
        }
        
        for timeframe, info in limits_info.items():
            max_days = info['max_days']
            
            # 안전 마진을 위해 95% 정도만 사용 (더 긴 백테스팅을 위해)
            safe_max_days = int(max_days * 0.95)
            
            config['timeframe_limits'][timeframe] = {
                'max_days': safe_max_days,
                'max_candles': int(info['max_candles'] * 0.95)
            }
            
            # 추천 기간 설정
            if timeframe in ['1m', '5m']:
                recommended = min(safe_max_days, 7)  # 최대 1주일
            elif timeframe in ['15m', '1h']:
                recommended = min(safe_max_days, 30)  # 최대 1개월
            elif timeframe == '4h':
                recommended = min(safe_max_days, 90)  # 최대 3개월
            elif timeframe == '1d':
                recommended = min(safe_max_days, 365)  # 최대 1년
                
            config['recommended_periods'][timeframe] = recommended
        
        print("✅ UI 제약 조건 설정 완료!")
        return config
    
    def run_full_test(self):
        """전체 테스트 실행"""
        print("🚀 Bitget API 제한 전체 테스트 시작!")
        print("=" * 60)
        
        # 1. 시간 간격별 제한 테스트
        limits_info = self.test_all_timeframes_limits()
        
        print("\n" + "=" * 60)
        
        # 2. API 속도 제한 테스트
        rate_limits = self.test_api_rate_limits()
        
        print("\n" + "=" * 60)
        
        # 3. UI 제약 조건 생성
        ui_config = self.generate_ui_constraints_config(limits_info)
        
        # 4. 결과 요약
        print("\n📋 **테스트 결과 요약**")
        print("=" * 60)
        
        for timeframe, info in limits_info.items():
            safe_days = ui_config['timeframe_limits'][timeframe]['max_days']
            recommended = ui_config['recommended_periods'][timeframe]
            
            print(f"🕐 {timeframe:>4}: 최대 {info['max_days']:>4.0f}일 → 안전 {safe_days:>3}일 (추천: {recommended:>3}일)")
        
        print(f"\n⚡ API 속도: {rate_limits['requests_per_second']:.1f} req/sec")
        
        return {
            'limits_info': limits_info,
            'rate_limits': rate_limits,
            'ui_config': ui_config
        }

def main():
    """메인 테스트 실행"""
    tester = BitgetAPILimitsTester()
    results = tester.run_full_test()
    
    # 설정 파일로 저장
    import json
    with open('api_limits_config.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 테스트 결과가 'api_limits_config.json'에 저장되었습니다.")

if __name__ == "__main__":
    main()