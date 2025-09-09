-- milvus_ddl.sql
-- Milvus 2.x: 컬렉션/인덱스 정의 스크립트 (SQL-like)
-- 실행: Attu의 SQL 콘솔 또는 SDK의 client.sql(...) 지원, 혹은 REST/CLI로 동등 작업 수행

-- 1) 데이터베이스 선택/생성
CREATE DATABASE IF NOT EXISTS cctv;
USE cctv;

-- 2) 기존 컬렉션 제거(있다면)
DROP COLLECTION IF EXISTS objects_vec;

-- 3) 컬렉션 생성
--   - id: PK (auto_id 사용)
--   - embedding: 512차원 벡터
--   - camera_id, ts, cls, track_id: 스칼라 메타데이터
--   - partition_key_field=camera_id 로 파티션 키 사용(카메라별 검색 범위 축소)
CREATE COLLECTION objects_vec (
  id         INT64 PRIMARY KEY,
  camera_id  INT64,
  ts         INT64,
  cls        VARCHAR(16),
  track_id   INT64,
  embedding  FLOAT_VECTOR[512]
)
WITH (
  auto_id = true,
  enable_dynamic_field = false,
  -- (선택) 파티션 키: camera_id 기준 자동 파티션
  partition_key_field = 'camera_id',
  num_partitions = 8,
  num_shards = 1,
  consistency_level = 'Bounded'
);

-- 4) 벡터 인덱스 (둘 중 하나 선택)
-- (A) IVF_FLAT + IP: 대용량에 유리, nlist로 리콜/지연 트레이드오프
CREATE INDEX idx_objects_vec_ivf
ON objects_vec (embedding)
USING IVF_FLAT
WITH (metric_type = 'IP', nlist = 1024);

-- (B) HNSW + IP: 온라인 질의 지연 안정성, efConstruction은 빌드·메모리 비용 증가
-- DROP INDEX IF EXISTS idx_objects_vec_ivf; -- IVF 대신 HNSW를 쓸 경우
-- CREATE INDEX idx_objects_vec_hnsw
-- ON objects_vec (embedding)
-- USING HNSW
-- WITH (metric_type = 'IP', M = 16, efConstruction = 200);

-- 5) 스칼라 인덱스(선택) — 필터 자주 쓰는 필드에 권장
-- cls 텍스트 필드에 Inverted 인덱스
CREATE INDEX idx_objects_cls
ON objects_vec (cls)
USING INVERTED;

-- ts 정렬/범위 필터 최적화(선택)
CREATE INDEX idx_objects_ts
ON objects_vec (ts)
USING STL_SORT;

-- 6) 서빙 로드
LOAD COLLECTION objects_vec;

-- 참고:
-- - DML/검색은 SDK 또는 milvus_cli 사용 권장
--   예: Python SDK로 insert/search 수행 (Milvus 문서의 Insert/Search 가이드 참조)
-- - 필터 결합 검색:
--   filter 예시는 "cls == 'person' && camera_id in [1,2] && ts >= 1725718800 && ts < 1725722400"
--   (벡터 검색과 함께 하이브리드로 사용) 
