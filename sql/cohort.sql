-- The PULSE sepsis cohort: ICU stays meeting Sepsis-3 (SOFA of at least 2
-- with suspected infection) with an emergency department stay and a
-- documented pre-ICU fluid volume. The intended exclusion of cardiac,
-- vascular and trauma surgery is written as a filter on the admission type,
-- which matches no MIMIC-IV admission type and so excludes no stay.
--
-- One DuckDB query that returns the subject, admission and stay of every
-- cohort stay and changes nothing in the database. Sepsis-3 is the concept
-- mimiciv_derived.sepsis3 of the MIMIC Code Repository
-- (github.com/MIT-LCP/mimic-code, mimic-iv/concepts_duckdb at release
-- v3.0.1). `python -m pulse build-cache` runs this query; on its own:
--
--   duckdb -readonly mimiciv.duckdb < sql/cohort.sql

WITH sepsis3_stays AS (
    SELECT DISTINCT s.subject_id, s.stay_id, i.hadm_id
    FROM mimiciv_derived.sepsis3 s
    JOIN mimiciv_icu.icustays i ON s.stay_id = i.stay_id
    WHERE s.sepsis3 = true
),
-- Fluids charted as given before admission or outside the ICU.
preicu_fluids AS (
    SELECT ss.stay_id, COALESCE(SUM(ie.amount), 0) AS preicu_fluid_volume
    FROM sepsis3_stays ss
    LEFT JOIN mimiciv_icu.inputevents ie ON ie.stay_id = ss.stay_id
        AND ie.ordercategoryname = '16-Pre Admission/Non-ICU'
        AND ie.amount > 0
    GROUP BY ss.stay_id
)
SELECT ss.subject_id, ss.hadm_id, ss.stay_id
FROM sepsis3_stays ss
JOIN mimiciv_hosp.admissions adm ON ss.hadm_id = adm.hadm_id
JOIN preicu_fluids pf ON ss.stay_id = pf.stay_id
WHERE
    -- No MIMIC-IV admission type contains these words, so they exclude no stay.
    LOWER(adm.admission_type) NOT LIKE '%cardiac%'
    AND LOWER(adm.admission_type) NOT LIKE '%vascular%'
    AND LOWER(adm.admission_type) NOT LIKE '%trauma%'
    AND adm.edregtime IS NOT NULL
    AND pf.preicu_fluid_volume > 0
ORDER BY ss.subject_id, ss.stay_id;
