#!/bin/bash

echo "======================================"
echo "  EdgeTTS API curl 使用範例"
echo "======================================"
echo ""

# API 服務地址
API_URL="http://localhost:8765"

echo "範例 1: 基本的 TTS 請求 (返回完整音檔)"
echo "----------------------------------------"
echo "curl -X POST \"$API_URL/tts\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"text\":\"你好,這是測試\",\"voice\":\"zh-TW-HsiaoChenNeural\"}' \\"
echo "  -o output.mp3"
echo ""

echo "範例 2: TTS 串流模式 (推薦用於長文本)"
echo "----------------------------------------"
echo "curl -X POST \"$API_URL/tts/stream\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"text\":\"歡迎使用 EdgeTTS 語音合成服務\",\"voice\":\"zh-TW-HsiaoChenNeural\"}' \\"
echo "  -o stream_output.mp3"
echo ""

echo "範例 3: 調整語速、音量和音調"
echo "----------------------------------------"
echo "curl -X POST \"$API_URL/tts\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"text\":\"快速朗讀測試\",\"voice\":\"zh-TW-HsiaoChenNeural\",\"rate\":\"+50%\",\"volume\":\"+20%\",\"pitch\":\"+10Hz\"}' \\"
echo "  -o fast_speech.mp3"
echo ""

echo "範例 4: 使用男聲"
echo "----------------------------------------"
echo "curl -X POST \"$API_URL/tts\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"text\":\"這是男聲測試\",\"voice\":\"zh-TW-YunJheNeural\"}' \\"
echo "  -o male_voice.mp3"
echo ""

echo "範例 5: 英文語音"
echo "----------------------------------------"
echo "curl -X POST \"$API_URL/tts\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"text\":\"Hello, this is a test\",\"voice\":\"en-US-JennyNeural\"}' \\"
echo "  -o english.mp3"
echo ""

echo "範例 6: 查看所有可用語音"
echo "----------------------------------------"
echo "curl -X GET \"$API_URL/voices\" | jq '.voices[] | {name, gender, locale}'"
echo ""

echo "範例 7: 查看台灣繁體中文語音"
echo "----------------------------------------"
echo "curl -X GET \"$API_URL/voices/zh-TW\""
echo ""

echo "範例 8: 串流模式長文本測試"
echo "----------------------------------------"
echo "curl -X POST \"$API_URL/tts/stream\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{\"text\":\"這是一段比較長的文字,用來測試串流模式的效果。串流模式可以在接收到第一個音訊片段時就開始播放,不需要等待整個音檔生成完畢,這樣可以提供更好的使用者體驗。\",\"voice\":\"zh-TW-HsiaoChenNeural\"}' \\"
echo "  -o long_stream.mp3"
echo ""

echo ""
echo "======================================"
echo "  實際執行範例"
echo "======================================"
echo ""

# 選擇要執行的範例
read -p "選擇要執行的範例 (1-8) 或按 Enter 跳過: " choice

case $choice in
    1)
        echo "執行範例 1..."
        curl -X POST "$API_URL/tts" \
          -H "Content-Type: application/json" \
          -d '{"text":"你好,這是測試","voice":"zh-TW-HsiaoChenNeural"}' \
          -o output.mp3
        echo "✅ 已儲存為 output.mp3"
        ;;
    2)
        echo "執行範例 2 (串流模式)..."
        curl -X POST "$API_URL/tts/stream" \
          -H "Content-Type: application/json" \
          -d '{"text":"歡迎使用 EdgeTTS 語音合成服務","voice":"zh-TW-HsiaoChenNeural"}' \
          -o stream_output.mp3
        echo "✅ 已儲存為 stream_output.mp3"
        ;;
    3)
        echo "執行範例 3 (調整參數)..."
        curl -X POST "$API_URL/tts" \
          -H "Content-Type: application/json" \
          -d '{"text":"快速朗讀測試","voice":"zh-TW-HsiaoChenNeural","rate":"+50%","volume":"+20%","pitch":"+10Hz"}' \
          -o fast_speech.mp3
        echo "✅ 已儲存為 fast_speech.mp3"
        ;;
    4)
        echo "執行範例 4 (男聲)..."
        curl -X POST "$API_URL/tts" \
          -H "Content-Type: application/json" \
          -d '{"text":"這是男聲測試","voice":"zh-TW-YunJheNeural"}' \
          -o male_voice.mp3
        echo "✅ 已儲存為 male_voice.mp3"
        ;;
    5)
        echo "執行範例 5 (英文)..."
        curl -X POST "$API_URL/tts" \
          -H "Content-Type: application/json" \
          -d '{"text":"Hello, this is a test","voice":"en-US-JennyNeural"}' \
          -o english.mp3
        echo "✅ 已儲存為 english.mp3"
        ;;
    6)
        echo "執行範例 6 (查看所有語音)..."
        curl -X GET "$API_URL/voices" | python3 -m json.tool | head -50
        ;;
    7)
        echo "執行範例 7 (台灣語音)..."
        curl -X GET "$API_URL/voices/zh-TW" | python3 -m json.tool
        ;;
    8)
        echo "執行範例 8 (長文本串流)..."
        curl -X POST "$API_URL/tts/stream" \
          -H "Content-Type: application/json" \
          -d '{"text":"這是一段比較長的文字,用來測試串流模式的效果。串流模式可以在接收到第一個音訊片段時就開始播放,不需要等待整個音檔生成完畢,這樣可以提供更好的使用者體驗。","voice":"zh-TW-HsiaoChenNeural"}' \
          -o long_stream.mp3
        echo "✅ 已儲存為 long_stream.mp3"
        ;;
    *)
        echo "已取消執行"
        ;;
esac

echo ""
echo "======================================"
echo "  快速測試指令"
echo "======================================"
echo ""
echo "# 基本測試 (直接複製使用):"
echo "curl -X POST 'http://localhost:8765/tts/stream' -H 'Content-Type: application/json' -d '{\"text\":\"測試語音\",\"voice\":\"zh-TW-HsiaoChenNeural\"}' -o test.mp3 && mpg123 test.mp3"
echo ""
echo "# 或使用 ffplay 播放:"
echo "curl -X POST 'http://localhost:8765/tts/stream' -H 'Content-Type: application/json' -d '{\"text\":\"即時播放測試\",\"voice\":\"zh-TW-HsiaoChenNeural\"}' | ffplay -nodisp -autoexit -"
echo ""
