-- PharmaPath AI — PostgreSQL Schema
-- Используется при продакшн-миграции (пока MVP работает на CSV)

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Doctors ──────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS doctors_base (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    full_name       VARCHAR(200) NOT NULL,
    gender          VARCHAR(1),
    specialty       VARCHAR(50) NOT NULL,
    category        CHAR(1) NOT NULL CHECK (category IN ('A','B','C')),
    work_address    TEXT,
    geo_location    GEOMETRY(Point, 4326),
    schedule_json   JSONB DEFAULT '{}',
    loyalty_score   NUMERIC(3,1) DEFAULT 5.0,
    avg_sales_brick NUMERIC(8,2) DEFAULT 0.0,
    phone           VARCHAR(20),
    email           VARCHAR(100),
    created_at      TIMESTAMP DEFAULT NOW(),
    updated_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_doctors_specialty ON doctors_base(specialty);
CREATE INDEX idx_doctors_category  ON doctors_base(category);
CREATE INDEX idx_doctors_geo       ON doctors_base USING GIST(geo_location);

-- ── Visits ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS visits_log (
    id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    doctor_id         UUID REFERENCES doctors_base(id),
    rep_id            VARCHAR(20) NOT NULL,
    visit_date        DATE NOT NULL,
    visit_time        TIME,
    day_of_week       SMALLINT,
    status            VARCHAR(20) NOT NULL,
    duration_minutes  SMALLINT DEFAULT 0,
    report_text       TEXT,
    report_parsed     JSONB,
    created_at        TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_visits_doctor  ON visits_log(doctor_id);
CREATE INDEX idx_visits_rep     ON visits_log(rep_id);
CREATE INDEX idx_visits_date    ON visits_log(visit_date DESC);

-- ── Routes ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS daily_routes (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rep_id          VARCHAR(20) NOT NULL,
    route_date      DATE NOT NULL,
    route_json      JSONB NOT NULL,
    total_score     NUMERIC(8,2),
    total_distance  NUMERIC(6,2),
    optimizer_status VARCHAR(20),
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_routes_rep_date ON daily_routes(rep_id, route_date DESC);

-- ── Reps ─────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS medical_reps (
    id          VARCHAR(20) PRIMARY KEY,
    full_name   VARCHAR(200) NOT NULL,
    territory   VARCHAR(50),
    home_lat    NUMERIC(9,6),
    home_lon    NUMERIC(9,6)
);

-- ── Spatial helper ───────────────────────────────────────────────────────────

-- Найти врачей в радиусе N метров
-- SELECT * FROM doctors_base
-- WHERE ST_DWithin(
--     geo_location,
--     ST_SetSRID(ST_MakePoint(37.6173, 55.7558), 4326)::geography,
--     3000  -- 3 km
-- );