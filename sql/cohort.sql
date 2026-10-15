-- The PULSE sepsis cohort: ICU stays meeting Sepsis-3 (SOFA of at least 2
-- with suspected infection) with an emergency department stay and a
-- documented pre-ICU fluid volume. The intended exclusion of cardiac,
-- vascular and trauma surgery is written as a filter on the admission type,
-- which matches no MIMIC-IV admission type and so excludes no stay.
-- Recomputes mimiciv_derived.sepsis3 from mimiciv_derived.sofa and
-- mimiciv_derived.suspicion_of_infection, then stores the cohort as
-- 'GT-Sepsis-Paper' in public.d_cohorts and public.cohort, creating these
-- two tables when they are absent.
--
-- The Sepsis-3 query is the concept of the MIMIC Code Repository
-- (github.com/MIT-LCP/mimic-code),
-- mimic-iv/concepts_postgres/sepsis/sepsis3.sql at release v3.0.0,
-- unchanged through v3.0.1, written into the existing table (TRUNCATE and
-- INSERT); the selection is the same.
-- Licence: MIT, Copyright (c) 2019 MIT Laboratory for Computational
-- Physiology; the full text is in sql/LICENSE-mimic-code.
--
-- WARNING: this script changes the MIMIC-IV database it runs on. It replaces the
-- contents of mimiciv_derived.sepsis3, and adds indexes on
-- mimiciv_icu.inputevents (stay_id) and mimiciv_derived.sepsis3.
-- Other projects that read these tables from the same database will see the
-- versions used for the paper. Run it on a database whose derived concepts
-- nothing else depends on, or rebuild the concepts afterwards.
--
--   psql -d mimiciv -f sql/cohort.sql

CREATE TABLE IF NOT EXISTS public.d_cohorts (
    cohort_id SERIAL PRIMARY KEY,
    cohort_name VARCHAR UNIQUE,
    cohort_description TEXT
);
CREATE TABLE IF NOT EXISTS public.cohort (
    cohort_id INTEGER,
    subject_id INTEGER,
    hadm_id INTEGER,
    stay_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_inputevents_stay_id ON mimiciv_icu.inputevents (stay_id);

TRUNCATE TABLE mimiciv_derived.sepsis3;
INSERT INTO mimiciv_derived.sepsis3
WITH sofa AS (
  SELECT
    stay_id,
    starttime,
    endtime,
    respiration_24hours AS respiration,
    coagulation_24hours AS coagulation,
    liver_24hours AS liver,
    cardiovascular_24hours AS cardiovascular,
    cns_24hours AS cns,
    renal_24hours AS renal,
    sofa_24hours AS sofa_score
  FROM mimiciv_derived.sofa
  WHERE sofa_24hours >= 2
), s1 AS (
  SELECT
    soi.subject_id,
    soi.stay_id,
    soi.ab_id,
    soi.antibiotic,
    soi.antibiotic_time,
    soi.culture_time,
    soi.suspected_infection,
    soi.suspected_infection_time,
    soi.specimen,
    soi.positive_culture,
    starttime,
    endtime,
    respiration,
    coagulation,
    liver,
    cardiovascular,
    cns,
    renal,
    sofa_score,
    (sofa_score >= 2 AND suspected_infection = 1) AS sepsis3,
    ROW_NUMBER() OVER (
      PARTITION BY soi.stay_id
      ORDER BY suspected_infection_time NULLS FIRST,
               antibiotic_time NULLS FIRST,
               culture_time NULLS FIRST,
               endtime NULLS FIRST
    ) AS rn_sus
  FROM mimiciv_derived.suspicion_of_infection AS soi
  INNER JOIN sofa ON soi.stay_id = sofa.stay_id
    AND sofa.endtime >= soi.suspected_infection_time - INTERVAL '48 HOUR'
    AND sofa.endtime <= soi.suspected_infection_time + INTERVAL '24 HOUR'
  WHERE soi.stay_id IS NOT NULL
)
SELECT
  subject_id,
  stay_id,
  antibiotic_time,
  culture_time,
  suspected_infection_time,
  endtime AS sofa_time,
  sofa_score,
  respiration,
  coagulation,
  liver,
  cardiovascular,
  cns,
  renal,
  sepsis3
FROM s1
WHERE rn_sus = 1;

CREATE INDEX IF NOT EXISTS idx_sepsis3_stay_id ON mimiciv_derived.sepsis3 (stay_id);
CREATE INDEX IF NOT EXISTS idx_sepsis3_subject_id ON mimiciv_derived.sepsis3 (subject_id);

DELETE FROM public.cohort WHERE cohort_id IN (
    SELECT cohort_id FROM public.d_cohorts WHERE cohort_name = 'GT-Sepsis-Paper'
);
DELETE FROM public.d_cohorts WHERE cohort_name = 'GT-Sepsis-Paper';
INSERT INTO public.d_cohorts (cohort_name, cohort_description)
VALUES ('GT-Sepsis-Paper', 'PULSE sepsis cohort: Sepsis-3, no cardiac, vascular or trauma admissions, ED stay, documented pre-ICU fluids');

INSERT INTO public.cohort (cohort_id, subject_id, stay_id, hadm_id)
WITH sepsis3_stays AS (
    SELECT DISTINCT s.subject_id, s.stay_id, i.hadm_id, i.intime
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
SELECT
    (SELECT cohort_id FROM public.d_cohorts WHERE cohort_name = 'GT-Sepsis-Paper'),
    ss.subject_id,
    ss.stay_id,
    ss.hadm_id
FROM sepsis3_stays ss
JOIN mimiciv_hosp.admissions adm ON ss.hadm_id = adm.hadm_id
JOIN preicu_fluids pf ON ss.stay_id = pf.stay_id
WHERE
    -- No MIMIC-IV admission type contains these words, so they exclude no stay.
    LOWER(adm.admission_type) NOT LIKE '%cardiac%'
    AND LOWER(adm.admission_type) NOT LIKE '%vascular%'
    AND LOWER(adm.admission_type) NOT LIKE '%trauma%'
    AND adm.edregtime IS NOT NULL
    AND pf.preicu_fluid_volume > 0;
