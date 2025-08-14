<img width="2100" height="1181" alt="image" src="https://github.com/user-attachments/assets/200a43e6-6d64-4e29-994e-0bc0b1328fc4" />

## KB AI Challenge — 타이핑 서명 본인 인증 솔루션

### 개요
- **목표**: 타이핑 서명을 이용해 사용자를 등록/인증하는 시스템
- **구성**: 안드로이드 앱 + FastAPI 백엔드(웹 UI 포함) + 모델 추론/암호화/DB
- **핵심 기능**:
  - 사용자 등록(여러 세션 평균 임베딩)
  - 사용자 인증/식별(L2 또는 코사인 거리, 임계값 자동 로드)
  - 임베딩 AES-GCM 암호화 저장(MySQL)

```
  [Client: Web/App] <-- JSON/API --> [Backend: FastAPI Server] <-- SQLAlchemy --> [Database: MySQL]
      |                                      |                                      ^
      |                                      v                                      |
      +----------------------------> [AI Model: PyTorch] ---------------------------+
```
---

### 레포지토리 구조
- `typing-signature-app/`: 안드로이드 앱(Compose UI, WebView 기반 타이핑 수집)
  - `app/src/main/java/com/hyunsung/key_stroke/data/Constants.kt`: 서버 URL 설정
  - `app/src/main/assets/`: HTML/JS 자산
  - `model-db/`: 파이썬 모델/서버(파일 기반 등록 모드)
- `typing-signature-web/`: FastAPI 백엔드(웹 데모 + DB 기반 등록/인증)
  - `web/server.py`: API/정적 페이지 서빙
  - `utils/db.py`: MySQL 접속/스키마
  - `utils/crypto.py`: 임베딩 암복호화(AES-GCM)
  - `docker-compose.yml`: MySQL + FastAPI 구동
- `typing-signature-model/`: 모델/평가 스켈레톤

---

## API 명세
- `GET /api/config`: `{ seq_len, threshold, users }`
- `GET /api/users`: `{ users: string[] }`
- `DELETE /api/users`: `{ deleted, users }` (모두 삭제)
- `DELETE /api/users/{user}`: `{ deleted, users }`
- `POST /api/enroll`
  - 요청: `{ user: string, sessions: List<List<Event>>, metric?: "l2"|"cosine" }`
  - `Event`: `{ press: number, release: number, key: string }` (초 단위)
  - 응답: `{ ok: true, user }`
- `POST /api/verify`
  - 요청: `{ user: string, events: List<Event>, metric?: "l2"|"cosine" }`
  - 응답: `{ distance: number, threshold?: number, decision?: "ACCEPT"|"REJECT" }`
- `POST /api/identify`
  - 요청: `{ events: List<Event>, metric?: "l2"|"cosine", topk?: number }`
  - 응답: `{ top: [{ user, distance }], threshold?: number, decision?: string }`
