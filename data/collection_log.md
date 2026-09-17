# Data Collection Log

## Purpose

This log records the manual data collection process for the Sydney Housing Price Prediction project (Task 8.1). Each entry documents a collection session, including date, number of records added, source, and any judgment calls made.

---

## Collection Summary

| Suburb | Target | Collected | Shortfall |
|--------|--------|-----------|-----------|
| Surry Hills | 40 | ___ | ___ |
| Parramatta | 40 | ___ | ___ |
| Penrith | 40 | ___ | ___ |
| **Total** | **120** | **___** | **___** |

---

## Session Log

### Session 1 — [Date]
- **Duration:** [X hours]
- **Source:** domain.com.au / realestate.com.au
- **Records added:** [N]
- **Suburbs covered:** [...]
- **Notes:**
  - [E.g., Land size missing for all Surry Hills units — recorded as NA]
  - [E.g., Several listings showed "price withheld" — excluded per brief requirement]
  - [E.g., Some sale dates were not publicly visible — recorded as NA]

### Session 2 — [Date]
- **Duration:** [X hours]
- **Records added:** [N]
- **Notes:**

### Session 3 — [Date]
- **Duration:** [X hours]
- **Records added:** [N]
- **Notes:**

---

## Judgment Calls

| Issue | Decision | Rationale |
|-------|----------|-----------|
| Units in Surry Hills — land size not listed | Record as `NA` | Units share communal land; the per-dwelling allocation is not meaningful |
| Floor area not listed on listing page | Record as `NA` | Do not estimate or fill from other sources |
| Price listed as a range (e.g., $900K–$950K) | Exclude | Cannot assign a definitive sale price |
| "Price withheld" listings | Exclude | No target variable available |
| Properties sold at auction, result not posted | Exclude | No confirmed sale price |
| Properties in postcodes adjacent to target suburb | Exclude | Reduces geographic consistency |
| Year built not listed | Record as `NA` | Will be imputed in the pipeline |
| Car spaces listed as "garage" (count unclear) | Record 1 | Conservative estimate |

---

## Data Quality Summary

**Key limitations identified during collection:**
1. Land size is structurally missing for most units
2. Floor area is inconsistently reported across agents
3. Sale prices may lag listing by several weeks; ensure date recorded is the settlement date, not listing date
4. Sample is restricted to publicly listed sold properties — off-market and private treaty sales are excluded

These limitations are documented in the report's data quality discussion (Part 1).
