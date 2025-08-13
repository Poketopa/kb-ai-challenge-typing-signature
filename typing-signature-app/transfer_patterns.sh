#!/bin/bash

# 타이핑 패턴 파일을 안드로이드 기기에서 맥북으로 전송하는 스크립트

echo "📱 안드로이드 기기에서 패턴 파일을 가져오는 중..."

# 안드로이드 기기에서 Downloads 폴더의 파일을 맥북 프로젝트 폴더로 전송
adb pull /storage/emulated/0/Download/typing_patterns.md /Users/lhs/Desktop/github/key-stroke-test/typing_patterns.md

if [ $? -eq 0 ]; then
    echo "✅ 파일 전송 성공!"
    echo "📁 저장 위치: /Users/lhs/Desktop/github/key-stroke-test/typing_patterns.md"
    
    # 파일 내용 미리보기
    echo ""
    echo "📄 파일 내용 미리보기:"
    echo "========================"
    tail -20 /Users/lhs/Desktop/github/key-stroke-test/typing_patterns.md
else
    echo "❌ 파일 전송 실패!"
    echo "안드로이드 기기가 연결되어 있는지 확인하세요."
fi 