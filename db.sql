-- db.sql : frame + object (location JSONB: {"x":..,"y":..,"z":..})
-- 실행 예: psql -U obuser -h localhost -d db -f db.sql

-- 1) 프레임 메타 테이블
CREATE TABLE IF NOT EXISTS frame (
  frame_id     BIGSERIAL PRIMARY KEY,
  object_count INTEGER     NOT NULL DEFAULT 0 CHECK (object_count >= 0),
  captured_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
  file_path    TEXT        NOT NULL
);

-- 조회 인덱스
CREATE INDEX IF NOT EXISTS idx_frame_captured_at ON frame (captured_at DESC);
CREATE INDEX IF NOT EXISTS idx_frame_object_count ON frame (object_count);

-- 2) 객체 테이블 (바운딩박스 기준 정규화 좌표를 JSONB로 저장)
--   location 예시: {"x":0.32, "y":0.70, "z":0.15}
--   - x,y : bbox 중심점 (0~1)
--   - z   : bbox 면적 비율(0~1) = ((x2-x1)*(y2-y1))/(W*H)
CREATE TABLE IF NOT EXISTS object (
  object_id    BIGSERIAL PRIMARY KEY,
  frame_id     BIGINT     NOT NULL REFERENCES frame(frame_id) ON DELETE CASCADE,
  object_type  VARCHAR(64) NOT NULL,
  confidence   REAL        NOT NULL CHECK (confidence >= 0.0 AND confidence <= 1.0),
  location     JSONB       NOT NULL,
  vector_id    TEXT,
  saved_at     TIMESTAMPTZ NOT NULL DEFAULT now(),

  -- location 필수 키/타입/범위 검증
  CONSTRAINT object_location_json_chk CHECK (
    jsonb_typeof(location) = 'object'
    AND jsonb_typeof(location->'x') = 'number'
    AND jsonb_typeof(location->'y') = 'number'
    AND jsonb_typeof(location->'z') = 'number'
    AND ((location->>'x')::double precision BETWEEN 0.0 AND 1.0)
    AND ((location->>'y')::double precision BETWEEN 0.0 AND 1.0)
    AND ((location->>'z')::double precision BETWEEN 0.0 AND 1.0)
  )
);

-- 조회/검색 인덱스
CREATE INDEX IF NOT EXISTS idx_object_saved_at     ON object (saved_at DESC);
CREATE INDEX IF NOT EXISTS idx_object_frame_id     ON object (frame_id);
CREATE INDEX IF NOT EXISTS idx_object_object_type  ON object (object_type);
-- location JSONB 검색이 필요하면 GIN 인덱스
CREATE INDEX IF NOT EXISTS idx_object_location_gin ON object USING GIN (location);

-- 3) frame.object_count 자동 유지 트리거
CREATE OR REPLACE FUNCTION trg_object_count_sync()
RETURNS TRIGGER AS $$
BEGIN
  IF TG_OP = 'INSERT' THEN
    UPDATE frame SET object_count = object_count + 1 WHERE frame_id = NEW.frame_id;
    RETURN NEW;
  ELSIF TG_OP = 'DELETE' THEN
    UPDATE frame SET object_count = GREATEST(object_count - 1, 0) WHERE frame_id = OLD.frame_id;
    RETURN OLD;
  ELSIF TG_OP = 'UPDATE' THEN
    IF NEW.frame_id IS DISTINCT FROM OLD.frame_id THEN
      UPDATE frame SET object_count = GREATEST(object_count - 1, 0) WHERE frame_id = OLD.frame_id;
      UPDATE frame SET object_count = object_count + 1 WHERE frame_id = NEW.frame_id;
    END IF;
    RETURN NEW;
  END IF;
  RETURN NULL;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_object_count_insert ON object;
CREATE TRIGGER trg_object_count_insert
AFTER INSERT ON object
FOR EACH ROW EXECUTE FUNCTION trg_object_count_sync();

DROP TRIGGER IF EXISTS trg_object_count_delete ON object;
CREATE TRIGGER trg_object_count_delete
AFTER DELETE ON object
FOR EACH ROW EXECUTE FUNCTION trg_object_count_sync();

DROP TRIGGER IF EXISTS trg_object_count_update ON object;
CREATE TRIGGER trg_object_count_update
AFTER UPDATE OF frame_id ON object
FOR EACH ROW EXECUTE FUNCTION trg_object_count_sync();

-- 4) 샘플 (주석 해제 후 테스트 가능)
-- INSERT INTO frame (file_path) VALUES ('/var/data/frames/2025-09-07T15-00-00Z.jpg') RETURNING frame_id;
-- INSERT INTO object (frame_id, object_type, confidence, location, vector_id)
-- VALUES (
--   1, 'person', 0.92,
--   '{"x":0.32,"y":0.70,"z":0.15}'::jsonb,
--   'vec_abc123'
-- );
