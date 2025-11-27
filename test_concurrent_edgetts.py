#!/usr/bin/env python3
"""
EdgeTTS API 併發測試工具
測試 API 在多個同時請求下的表現
"""
import asyncio
import aiohttp
import time
from datetime import datetime

API_URL = "http://52.69.235.212:8765/tts"

async def make_request(session, request_id, text):
    """發送單個 TTS 請求"""
    start_time = time.time()
    
    payload = {
        "text": text,
        "voice": "zh-TW-HsiaoChenNeural",
        "format": "mp3"
    }
    
    try:
        async with session.post(API_URL, json=payload) as response:
            content = await response.read()
            elapsed = time.time() - start_time
            
            if response.status == 200:
                print(f"[✓] 請求 {request_id:2d} 成功 | 耗時: {elapsed:.2f}s | 大小: {len(content):,} bytes")
                return {"id": request_id, "success": True, "time": elapsed, "size": len(content)}
            else:
                print(f"[✗] 請求 {request_id:2d} 失敗 | 狀態碼: {response.status}")
                return {"id": request_id, "success": False, "time": elapsed}
    
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[✗] 請求 {request_id:2d} 異常 | 錯誤: {str(e)}")
        return {"id": request_id, "success": False, "time": elapsed, "error": str(e)}

async def test_concurrent_requests(num_requests, text):
    """測試並發請求"""
    print(f"\n{'='*70}")
    print(f"🚀 併發測試開始 - {num_requests} 個同時請求")
    print(f"{'='*70}")
    print(f"測試時間: {datetime.now()}")
    print(f"API 端點: {API_URL}")
    print(f"測試文字: {text}")
    print(f"{'='*70}\n")
    
    # 記錄總開始時間
    total_start = time.time()
    
    # 創建會話並發送所有請求
    async with aiohttp.ClientSession() as session:
        tasks = [
            make_request(session, i+1, text)
            for i in range(num_requests)
        ]
        
        # 並發執行所有請求
        results = await asyncio.gather(*tasks)
    
    # 計算總時間
    total_time = time.time() - total_start
    
    # 統計結果
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print(f"\n{'='*70}")
    print(f"📊 測試結果統計")
    print(f"{'='*70}")
    print(f"總請求數: {num_requests}")
    print(f"成功數量: {len(successful)} ({len(successful)/num_requests*100:.1f}%)")
    print(f"失敗數量: {len(failed)} ({len(failed)/num_requests*100:.1f}%)")
    print(f"總耗時: {total_time:.2f} 秒")
    
    if successful:
        times = [r["time"] for r in successful]
        sizes = [r["size"] for r in successful]
        
        print(f"\n⏱️  響應時間分析:")
        print(f"  - 最快: {min(times):.2f}s")
        print(f"  - 最慢: {max(times):.2f}s")
        print(f"  - 平均: {sum(times)/len(times):.2f}s")
        
        print(f"\n📦 數據大小分析:")
        print(f"  - 平均大小: {sum(sizes)/len(sizes):,.0f} bytes")
        print(f"  - 總下載: {sum(sizes):,} bytes ({sum(sizes)/1024/1024:.2f} MB)")
        
        print(f"\n🎯 性能指標:")
        print(f"  - 吞吐量: {len(successful)/total_time:.2f} 請求/秒")
        print(f"  - 並發效率: {(sum(times)/len(times))/total_time:.2f}x")
    
    print(f"{'='*70}\n")
    
    return results

async def main():
    """主測試函數"""
    # 測試文字
    test_text = "孟鄉生化科技提供一站式的保養品代工服務"
    
    # 測試不同的併發量
    test_scenarios = [
        {"num": 5, "desc": "低併發測試 (5 個請求)"},
        {"num": 10, "desc": "中併發測試 (10 個請求)"},
        {"num": 20, "desc": "高併發測試 (20 個請求)"},
    ]
    
    all_results = {}
    
    for scenario in test_scenarios:
        print(f"\n\n{'#'*70}")
        print(f"# {scenario['desc']}")
        print(f"{'#'*70}")
        
        results = await test_concurrent_requests(scenario['num'], test_text)
        all_results[scenario['num']] = results
        
        # 等待一下再進行下一組測試
        await asyncio.sleep(2)
    
    # 最終總結
    print(f"\n\n{'='*70}")
    print(f"🏁 全部測試完成")
    print(f"{'='*70}")
    print(f"\n建議:")
    print(f"  ✓ 如果所有測試都成功,說明 API 可以良好處理併發")
    print(f"  ✓ 如果高併發測試出現失敗,可以考慮增加 uvicorn workers")
    print(f"  ✓ 關注平均響應時間,如果隨併發量增加太多,需要優化")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                EdgeTTS API 併發性能測試工具                     ║
║                                                                  ║
║  此工具會測試 API 在不同併發量下的表現                          ║
║  包括: 5, 10, 20 個同時請求                                     ║
╚══════════════════════════════════════════════════════════════════╝
""")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n測試被用戶中斷")
    except Exception as e:
        print(f"\n\n測試過程中發生錯誤: {str(e)}")
