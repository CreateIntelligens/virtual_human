# server.py - Part 4: Request Handlers (追加)

async def health_check(request):
    """健康檢查端點，用於 Docker 容器的健康監控"""
    try:
        # 獲取 SRS 服務狀態，僅作為參考不影響結果
        srs_status = "unknown"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get('http://srs:1985/api/v1/versions', timeout=2) as resp:
                    srs_status = "up" if resp.status == 200 else "down"
        except:
            srs_status = "down"

        # 檢查應用服務狀態
        status = {
            "app": "healthy",
            "srs": srs_status,
            "sessions": len(nerfreals),
            "connections": len(pcs)
        }

        # 檢查會話數量
        if len(pcs) >= opt.max_session:
            status["app"] = "overloaded"
            return web.Response(
                status=503,  # Service Unavailable
                text=json.dumps(status),
                content_type='application/json'
            )

        # 檢查連接狀態
        broken_sessions = []
        for session_id, nerfreal in nerfreals.items():
            if nerfreal is None:
                broken_sessions.append(session_id)

        if broken_sessions:
            status["app"] = "degraded"
            status["broken_sessions"] = broken_sessions

        return web.Response(
            status=200,
            text=json.dumps(status),
            content_type='application/json'
        )

    except Exception as e:
        error_status = {
            "app": "error",
            "error": str(e)
        }
        logger.error(f"Health check failed: {e}")
        return web.Response(
            status=500,
            text=json.dumps(error_status),
            content_type='application/json'
        )

# 添加到 Part 4 的路由配置中（其餘路由保持不變）
# appasync.router.add_post("/offer", offer)
# appasync.router.add_post("/human", human)
# [... 其他現有路由 ...]
appasync.router.add_get("/health", health_check)  # 添加健康檢查端點
