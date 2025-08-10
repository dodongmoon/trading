import json
import math

def generate_3x_split_limits():
    """3회 분할을 고려한 새로운 제한 설정 생성"""
    
    # 기본 API 제한 정보 (1회당)
    base_limits = {
        "15m": {"max_candles": 1000, "minutes_per_candle": 15},
        "30m": {"max_candles": 1000, "minutes_per_candle": 30},
        "1h": {"max_candles": 1000, "minutes_per_candle": 60},
        "4h": {"max_candles": 540, "minutes_per_candle": 240},
        "1d": {"max_candles": 90, "minutes_per_candle": 1440}
    }
    
    # 3회 분할 + 95% 안전 마진 계산
    split_3x_limits = {}
    
    print("🔢 3회 분할 제한 계산 결과:")
    print("=" * 80)
    
    for timeframe, info in base_limits.items():
        # 1회당 95% 안전 마진
        safe_candles_per_call = int(info["max_candles"] * 0.95)
        
        # 3회 분할 총 봉수
        total_candles_3x = safe_candles_per_call * 3
        
        # 총 분 수 계산
        total_minutes = total_candles_3x * info["minutes_per_candle"]
        
        # 총 일수 계산 (소수점 버림)
        total_days = math.floor(total_minutes / (24 * 60))
        
        split_3x_limits[timeframe] = {
            "max_candles_per_call": safe_candles_per_call,
            "max_calls": 3,
            "total_max_candles": total_candles_3x,
            "total_max_days": total_days,
            "minutes_per_candle": info["minutes_per_candle"]
        }
        
        print(f"📊 {timeframe:4s} | "
              f"1회: {safe_candles_per_call:4d}개 | "
              f"3회: {total_candles_3x:4d}개 | "
              f"총 {total_minutes:5d}분 | "
              f"최대 {total_days:3d}일")
    
    print("=" * 80)
    
    # JSON 파일로 저장
    output_config = {
        "split_3x_limits": split_3x_limits,
        "generation_info": {
            "max_splits": 3,
            "safety_margin": 0.95,
            "description": "3회 분할 호출을 고려한 최대 백테스팅 기간 설정"
        }
    }
    
    with open('api_limits_3x_config.json', 'w', encoding='utf-8') as f:
        json.dump(output_config, f, indent=2, ensure_ascii=False)
    
    print("✅ 설정 파일 저장 완료: api_limits_3x_config.json")
    return output_config

if __name__ == "__main__":
    config = generate_3x_split_limits()
    
    print("\n🎯 UI에서 사용할 최대 기간:")
    for timeframe, limits in config["split_3x_limits"].items():
        print(f"   {timeframe}: 최대 {limits['total_max_days']}일 "
              f"({limits['total_max_candles']}개 봉)")