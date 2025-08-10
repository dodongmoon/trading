import ccxt
import pandas as pd
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import time
import json

# .env 파일에서 API 키 로드
load_dotenv()

class ExtendedBitgetDataCollector:
    def __init__(self):
        """확장된 Bitget 데이터 수집기 초기화"""
        self.exchange = ccxt.bitget({
            'apiKey': os.getenv("BITGET_API_KEY"),
            'secret': os.getenv("BITGET_API_SECRET"),
            'password': os.getenv("BITGET_PASSPHRASE"),
            'options': {
                'defaultType': 'swap',
                'adjustForTimeDifference': True,
            },
        })
        self.exchange.set_sandbox_mode(False)
        
        # API 제한 정보 로드
        self.load_api_limits()
        
    def load_api_limits(self):
        """API 제한 정보 로드 (3회 분할 지원)"""
        try:
            # 3회 분할 설정 로드
            with open('api_limits_3x_config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.limits_3x = config['split_3x_limits']
                
            # 기존 설정도 백업으로 로드
            with open('api_limits_config.json', 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.limits_legacy = config['ui_config']['timeframe_limits']
                
        except FileNotFoundError:
            # 기본 제한값 설정
            self.limits_3x = {
                '15m': {'max_candles_per_call': 950, 'total_max_days': 29, 'max_calls': 3},
                '30m': {'max_candles_per_call': 950, 'total_max_days': 59, 'max_calls': 3},
                '1h': {'max_candles_per_call': 950, 'total_max_days': 118, 'max_calls': 3},
                '4h': {'max_candles_per_call': 513, 'total_max_days': 256, 'max_calls': 3},
                '1d': {'max_candles_per_call': 85, 'total_max_days': 255, 'max_calls': 3}
            }
            self.limits_legacy = {
                '15m': {'max_candles': 950, 'max_days': 9},
                '30m': {'max_candles': 950, 'max_days': 19},
                '1h': {'max_candles': 950, 'max_days': 39},
                '4h': {'max_candles': 513, 'max_days': 85},
                '1d': {'max_candles': 85, 'max_days': 85}
            }
    
    def calculate_optimal_splits(self, timeframe, requested_days):
        """
        요청된 기간에 대한 최적 분할 계산 (최대 3회 제한)
        
        Args:
            timeframe (str): 시간 간격 ('15m', '30m', '1h', '4h', '1d' 등)
            requested_days (int): 요청된 일수
            
        Returns:
            list: 분할된 기간 리스트 [(days1, days2, ...)]
        """
        # 3회 분할 설정 사용
        limits = self.limits_3x.get(timeframe, {})
        max_calls = limits.get('max_calls', 3)
        total_max_days = limits.get('total_max_days', 30)
        
        # 요청된 기간이 최대 허용 기간을 초과하는 경우
        if requested_days > total_max_days:
            print(f"⚠️ 요청된 기간 {requested_days}일이 최대 허용 기간 {total_max_days}일을 초과합니다.")
            requested_days = total_max_days
        
        # 1회로 충분한 경우
        single_call_max = limits.get('max_candles_per_call', 950)
        
        # 시간간격별 1일당 봉수 계산
        minutes_per_day = 24 * 60
        if timeframe == '15m':
            candles_per_day = minutes_per_day // 15
        elif timeframe == '30m':
            candles_per_day = minutes_per_day // 30
        elif timeframe == '1h':
            candles_per_day = minutes_per_day // 60
        elif timeframe == '4h':
            candles_per_day = minutes_per_day // 240
        elif timeframe == '1d':
            candles_per_day = 1
        else:
            candles_per_day = 24  # 기본값 (1시간 기준)
        
        # 1회 호출로 가능한 최대 일수
        max_days_per_call = single_call_max // candles_per_day
        
        if requested_days <= max_days_per_call:
            return [requested_days]
        
        # 여러 회 분할 필요
        splits = []
        remaining_days = requested_days
        calls_used = 0
        
        while remaining_days > 0 and calls_used < max_calls:
            if remaining_days <= max_days_per_call:
                splits.append(remaining_days)
                break
            else:
                splits.append(max_days_per_call)
                remaining_days -= max_days_per_call
                calls_used += 1
        
        print(f"📊 분할 계획: {requested_days}일 → {splits} (총 {len(splits)}회 요청)")
        return splits
    
    def fetch_historical_data_extended(self, symbol='BTCUSDT', timeframe='1h', days=30):
        """
        확장된 역사적 데이터 수집 (API 제한 우회)
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 시간 간격
            days (int): 수집할 일수
            
        Returns:
            pd.DataFrame: 확장된 OHLCV 데이터
        """
        print(f"🚀 확장 데이터 수집 시작: {symbol} {timeframe} {days}일")
        print("=" * 60)
        
        # 최적 분할 계산
        splits = self.calculate_optimal_splits(timeframe, days)
        
        if len(splits) == 1:
            # 분할 불필요 - 기본 수집
            return self.fetch_single_period(symbol, timeframe, days)
        
        # 분할 수집
        all_dataframes = []
        current_end_time = datetime.now()
        
        for i, period_days in enumerate(splits):
            print(f"\n📥 {i+1}/{len(splits)} 구간 수집 중... ({period_days}일)")
            
            # 각 구간의 시작/끝 시간 계산
            period_start = current_end_time - timedelta(days=period_days)
            
            print(f"   기간: {period_start.strftime('%Y-%m-%d %H:%M')} ~ {current_end_time.strftime('%Y-%m-%d %H:%M')}")
            
            # 해당 구간 데이터 수집
            df_segment = self.fetch_single_period(
                symbol, 
                timeframe, 
                period_days, 
                end_time=current_end_time
            )
            
            if df_segment is not None and len(df_segment) > 0:
                all_dataframes.append(df_segment)
                print(f"   ✅ {len(df_segment)}개 봉 수집 완료")
                print(f"   📅 실제 기간: {df_segment.index[0]} ~ {df_segment.index[-1]}")
                
                # 다음 구간의 끝 시간을 현재 구간의 가장 오래된 시점으로 설정
                # 이렇게 하면 경계 봉이 자연스럽게 1개 중복되며, 나중에 중복 제거로 처리
                current_end_time = df_segment.index[0]
                print(f"   🔄 다음 구간 끝 시간: {current_end_time}")
            else:
                print(f"   ❌ 구간 {i+1} 수집 실패")
            
            # API 제한 방지를 위한 대기
            time.sleep(1)
        
        if not all_dataframes:
            print("❌ 모든 구간 수집 실패")
            return None
        
        # 모든 구간 데이터 합치기
        print(f"\n🔗 {len(all_dataframes)}개 구간 데이터 병합 중...")
        
        # 각 구간별 봉수 출력 (디버깅용)
        total_before_merge = sum(len(df) for df in all_dataframes)
        print(f"   📊 구간별 봉수: {[len(df) for df in all_dataframes]}")
        print(f"   📈 병합 전 총 봉수: {total_before_merge}개")
        
        # 시간순으로 정렬하여 합치기 (오래된 것부터)
        # 최근 구간부터 append 했기 때문에 뒤집어서 오래된 → 최신 순으로 정렬
        all_dataframes.reverse()
        combined_df = pd.concat(all_dataframes, axis=0)

        # 중복 제거 전 봉수
        before_dedup = len(combined_df)
        print(f"   🔄 concat 후 봉수: {before_dedup}개")

        # 인덱스(타임스탬프) 기준 중복 제거 후 시간 순 정렬
        # 동일 타임스탬프 중복 시 최신 구간 값을 보존(keep='last')
        # 이제 경계 봉이 의도적으로 1개씩 중복되므로 중복 제거가 정상 작동함
        combined_df = combined_df.sort_index()
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
        
        # 중복 제거 후 봉수
        after_dedup = len(combined_df)
        duplicates_removed = before_dedup - after_dedup
        
        print(f"✅ 확장 데이터 수집 완료!")
        print(f"   - 총 구간: {len(splits)}개")
        print(f"   - 중복 제거: {duplicates_removed}개 봉 제거")
        print(f"   - 최종 봉수: {after_dedup}개")
        print(f"   - 데이터 효율성: {(after_dedup/total_before_merge*100):.1f}%")
        print(f"   - 기간: {combined_df.index[0]} ~ {combined_df.index[-1]}")
        print(f"   - 실제 일수: {(combined_df.index[-1] - combined_df.index[0]).days}일")
        
        return combined_df
    
    def fetch_single_period(self, symbol, timeframe, days, end_time=None):
        """
        단일 기간 데이터 수집
        
        Args:
            symbol (str): 거래 심볼
            timeframe (str): 시간 간격
            days (int): 수집할 일수
            end_time (datetime): 끝 시간 (None이면 현재 시간)
            
        Returns:
            pd.DataFrame: OHLCV 데이터
        """
        try:
            if end_time is None:
                end_time = datetime.now()
            
            start_time = end_time - timedelta(days=days)
            since = int(start_time.timestamp() * 1000)
            
            # 한번에 가져올 수 있는 최대 캔들 수 (3회 분할 설정 사용)
            limits_3x = self.limits_3x.get(timeframe, {})
            max_candles = limits_3x.get('max_candles_per_call', 1000)
            
            all_ohlcv = []
            end_time_ms = int(end_time.timestamp() * 1000)
            start_time_ms = int(start_time.timestamp() * 1000)
            current_end_ms = end_time_ms

            # Bitget은 until/endTime 기준으로 "해당 시점까지의 마지막 N개"를 반환하므로
            # 과거로 내려가며 역방향 페이징을 수행한다.
            while current_end_ms >= start_time_ms:
                params = {'until': current_end_ms, 'endTime': current_end_ms}
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, None, max_candles, params)
                if not ohlcv:
                    print("   ⛔ fetch_ohlcv 응답이 비었습니다. 중단")
                    break

                first_ts = ohlcv[0][0]
                last_ts_raw = ohlcv[-1][0]
                print(
                    f"   🔎 페이지: until={current_end_ms} resp={len(ohlcv)}개 "
                    f"resp_range={datetime.fromtimestamp(first_ts/1000)}~{datetime.fromtimestamp(last_ts_raw/1000)}"
                )

                # 요청 범위 [start_time_ms, end_time_ms]로 필터링
                filtered_ohlcv = [c for c in ohlcv if start_time_ms <= c[0] <= end_time_ms]
                if not filtered_ohlcv:
                    # 더 과거로 진행
                    current_end_ms = first_ts - 1
                    print(f"   ↩️ 유효 구간 없음. 더 과거로 이동: new_until={current_end_ms}")
                    time.sleep(0.1)
                    continue

                all_ohlcv.extend(filtered_ohlcv)

                first_kept = filtered_ohlcv[0][0]
                last_kept = filtered_ohlcv[-1][0]
                print(
                    f"   ✅ 필터링 후 누적={len(all_ohlcv)}개, kept_range={datetime.fromtimestamp(first_kept/1000)}~{datetime.fromtimestamp(last_kept/1000)}"
                )

                # 시작 지점에 도달했으면 종료
                if first_ts <= start_time_ms:
                    print("   ✅ start_time 도달. 종료")
                    break

                # 다음 루프: 현재 페이지의 가장 오래된 ts 이전으로 이동
                current_end_ms = first_ts - 1
                print(f"   ⏮ 다음 until={current_end_ms}")
                time.sleep(0.1)
            
            if not all_ohlcv:
                return None
            
            # DataFrame으로 변환
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # 중복 제거 및 정렬
            df = df.drop_duplicates().sort_index()
            
            return df
            
        except Exception as e:
            print(f"❌ 단일 기간 데이터 수집 오류: {e}")
            return None

def test_extended_data_collection():
    """확장 데이터 수집 테스트"""
    collector = ExtendedBitgetDataCollector()
    
    print("🧪 확장 데이터 수집 테스트")
    print("=" * 60)
    
    # 테스트 케이스들
    test_cases = [
        {'timeframe': '1h', 'days': 30, 'description': '1시간봉 30일 (분할 불필요)'},
        {'timeframe': '1h', 'days': 66, 'description': '1시간봉 66일 (2분할 확인용)'},
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🔬 테스트 {i}: {test_case['description']}")
        print("-" * 40)
        
        df = collector.fetch_historical_data_extended('BTC/USDT:USDT', test_case['timeframe'], test_case['days'])
        
        if df is not None:
            start_ts, end_ts = df.index[0], df.index[-1]
            actual_days = (end_ts - start_ts).days
            print(f"✅ 성공: {len(df)}개 봉, 실제 {actual_days}일, 기간 {start_ts} ~ {end_ts}")
            
            # CSV 저장
            filename = f"extended_data_{test_case['timeframe']}_{test_case['days']}d.csv"
            df.to_csv(filename)
            print(f"💾 저장: {filename}")
        else:
            print("❌ 실패")
        
        print("\n" + "="*60)
        time.sleep(2)  # 테스트 간 대기

if __name__ == "__main__":
    test_extended_data_collection()