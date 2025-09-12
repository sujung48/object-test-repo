# object-test-repo

## 가상환경 설정
(venv/ 폴더는 Git에 포함되지 않으며, 각 사용자가 로컬에서 직접 생성해야 함)


    macOS   |   python3 -m venv venv1

3. 가상환경 이동
    macOS   |   source venv1/bin/activate 

4. 의존성 설치
    macOS   |   pip install -r requirements.txt

5. FastAPI 서버 실행
    Windows |   backend\start.bat (cmd)
                .\backend\start.ps1 (powershell)
    macOS   |   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000


## DB 설정

1. PostgreSQL 설치
- macOS: `brew install postgresql@16`

2. DB/계정 생성
psql -h localhost -d postgres   #(postgresql 접속)
CREATE DATABASE db;
CREATE USER obuser WITH PASSWORD '1111';
GRANT ALL PRIVILEGES ON DATABASE db TO obuser;

\c db
GRANT CREATE, USAGE ON SCHEMA public TO obuser;
ALTER SCHEMA public OWNER TO obuser;   #(실패해도 무방)
GRANT ALL ON DATABASE db TO obuser;
\q (postgresql 종료)

3. 스키마 파일 적용
psql -U obuser -h localhost -d db -f db.sql

4. DB 접속
psql -h localhost -U obuser -d db

5. DB 명령어 정리


############################################
curl -X POST http://localhost:8000/camera/start   
curl -X POST http://localhost:8000/save           


curl -s "http://localhost:8000/frames/recent?limit=5" | jq .
n번째 프레임에서 객체 n개가 탐지되어 저장된거 확인

curl -s "http://localhost:8000/objects/recent?limit=5" | jq .
제일 최근 프레임에서 탐지된 객체 정보 저장된거 확인

curl -s -X POST http://localhost:8000/milvus/upsert/latest | jq . 
가장 최근 저장된 프레임에서 탐지된 객체중 n개를 milvus에 저장하기

curl -s "http://localhost:8000/milvus/search?topk=5" | jq .
milvus에서 방금 넣은 객체를 검색했더니 자기 자신이 1등으로 나옴
더미 임베딩이라 항상 “자기 자신=1위, 거리≈0”만 확인되지만, 실제로 임베딩 모델을 붙이면 “비슷한 객체끼리 가까운 거리”
