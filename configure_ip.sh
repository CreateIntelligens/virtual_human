#!/bin/bash

# IP 地址配置腳本
# 使用方法: ./configure_ip.sh <YOUR_IP>

set -e

if [ -z "$1" ]; then
    echo "❌ 錯誤：請提供 IP 地址"
    echo ""
    echo "使用方法: $0 <YOUR_IP>"
    echo "例如: $0 192.168.1.100"
    echo "或者: $0 localhost"
    echo ""
    exit 1
fi

YOUR_IP=$1

echo "🔧 開始配置 IP 地址為: $YOUR_IP"
echo ""

# 檢查必要文件是否存在
echo "📋 檢查必要文件..."
required_files=(
    "avatar_voice.json"
    "streaming_tts/api_v3_live.py"
    "web/rtcpushapi_google_360_test.html"
    "environment.md"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -ne 0 ]; then
    echo "❌ 以下必要文件不存在："
    for file in "${missing_files[@]}"; do
        echo "   - $file"
    done
    echo ""
    echo "請確保在專案根目錄執行此腳本"
    exit 1
fi

echo "✅ 所有必要文件都存在"
echo ""

# 備份原始文件
echo "📁 備份原始配置文件..."
backup_dir="backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$backup_dir"

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        cp "$file" "$backup_dir/"
        echo "   ✓ 備份 $file"
    fi
done

echo "   📂 備份文件保存在: $backup_dir/"
echo ""

# 修改配置文件
echo "⚙️ 修改配置文件..."

# 1. 修改 avatar_voice.json
echo "   🔄 修改 avatar_voice.json..."
if sed -i.tmp "s/192\.168\.1\.100/$YOUR_IP/g" avatar_voice.json; then
    rm -f avatar_voice.json.tmp
    echo "   ✅ avatar_voice.json 修改完成"
else
    echo "   ❌ avatar_voice.json 修改失敗"
fi

# 2. 修改 streaming_tts/api_v3_live.py
echo "   🔄 修改 streaming_tts/api_v3_live.py..."
if sed -i.tmp "s/<OLD_DOMAIN>/$YOUR_IP/g" streaming_tts/api_v3_live.py; then
    rm -f streaming_tts/api_v3_live.py.tmp
    echo "   ✅ streaming_tts/api_v3_live.py 修改完成"
else
    echo "   ❌ streaming_tts/api_v3_live.py 修改失敗"
fi

# 3. 修改 web/rtcpushapi_google_360_test.html
echo "   🔄 修改 web/rtcpushapi_google_360_test.html..."
if sed -i.tmp "s/192\.168\.1\.100/$YOUR_IP/g" web/rtcpushapi_google_360_test.html; then
    rm -f web/rtcpushapi_google_360_test.html.tmp
    echo "   ✅ web/rtcpushapi_google_360_test.html 修改完成"
else
    echo "   ❌ web/rtcpushapi_google_360_test.html 修改失敗"
fi

# 4. 修改 environment.md
echo "   🔄 修改 environment.md..."
if sed -i.tmp "s/192\.168\.1\.100/$YOUR_IP/g" environment.md; then
    rm -f environment.md.tmp
    echo "   ✅ environment.md 修改完成"
else
    echo "   ❌ environment.md 修改失敗"
fi

echo ""

# 驗證修改結果
echo "🔍 驗證修改結果..."
echo ""

# 檢查是否還有未替換的模板 IP 地址
template_ip_count=$(grep -r "192\.168\.1\.100" avatar_voice.json environment.md web/rtcpushapi_google_360_test.html 2>/dev/null | wc -l)
template_domain_count=$(grep -r "<OLD_DOMAIN>" streaming_tts/api_v3_live.py 2>/dev/null | wc -l)

if [ "$template_ip_count" -eq 0 ] && [ "$template_domain_count" -eq 0 ]; then
    echo "✅ 所有模板 IP 地址和域名都已成功替換為: $YOUR_IP"
else
    echo "⚠️  警告：仍有部分模板 IP 地址或域名未替換"
    if [ "$template_ip_count" -gt 0 ]; then
        echo "   剩餘的 192.168.1.100: $template_ip_count 處"
        grep -rn "192\.168\.1\.100" avatar_voice.json environment.md web/rtcpushapi_google_360_test.html 2>/dev/null || true
    fi
    if [ "$template_domain_count" -gt 0 ]; then
        echo "   剩餘的 <OLD_DOMAIN>: $template_domain_count 處"
        grep -rn "<OLD_DOMAIN>" streaming_tts/api_v3_live.py 2>/dev/null || true
    fi
fi

echo ""

# 顯示修改摘要
echo "📝 修改摘要："
echo "   🎯 目標 IP: $YOUR_IP"
echo "   📁 備份目錄: $backup_dir/"
echo "   📄 修改的文件："
echo "      - avatar_voice.json (TTS 服務器地址)"
echo "      - streaming_tts/api_v3_live.py (API 地址)"
echo "      - web/rtcpushapi_google_360_test.html (前端 API 地址)"
echo "      - environment.md (示例 IP)"
echo ""

# 創建 .env 文件
echo "📄 創建 .env 文件..."
cat > .env << EOF
# 虛擬人系統環境配置
CANDIDATE=$YOUR_IP
TRANSPORT=rtcpush
MODEL=musetalk
AVATAR_ID=avator_1
TTS=gpt-sovits
TTS_SERVER=http://$YOUR_IP:9885
MAX_SESSION=5
AVATAR_LIST=avator_2,avator_10,avator_3

# OpenAI 配置（可選）
# OPENAI_API_KEY=your_openai_api_key
# OPENAI_TTS_MODEL=tts-1
# OPENAI_VOICE=alloy

# 其他配置
DEBUG=1
EOF

echo "   ✅ .env 文件已創建"
echo ""

# 提供下一步指引
echo "🚀 配置完成！下一步："
echo ""
echo "1. 📦 下載模型文件（如果尚未下載）："
echo "   chmod +x download_models.sh"
echo "   ./download_models.sh"
echo ""
echo "2. 🐳 啟動 Docker 服務："
echo "   docker-compose up -d"
echo ""
echo "3. 🌐 訪問系統："
echo "   http://$YOUR_IP:8010/rtcpushapi_google_360_test.html"
echo ""
echo "4. 📊 檢查服務狀態："
echo "   docker-compose ps"
echo "   docker-compose logs -f"
echo ""
echo "5. 🔧 測試服務："
echo "   curl http://$YOUR_IP:8010/health"
echo "   curl http://$YOUR_IP:9885/"
echo "   curl http://$YOUR_IP:1985/api/v1/versions"
echo ""

# 恢復指令
echo "💡 如需恢復原始配置："
echo "   cp $backup_dir/* ."
echo "   cp $backup_dir/streaming_tts/api_v3_live.py streaming_tts/"
echo "   cp $backup_dir/web/rtcpushapi_google_360_test.html web/"
echo ""

echo "✨ IP 配置完成！"
