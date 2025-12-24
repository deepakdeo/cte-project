# Collect Your Own Data

This guide explains how to collect and format behavioral data for use with CTE.

---

## Quick Start

1. **Use a spreadsheet** (Google Sheets, Excel) to track daily metrics
2. **Export as CSV** when ready
3. **Run the cleaning pipeline** to process your data

---

## Data Schema

### Required Columns

These columns are essential for core functionality:

| Column | Format | Example | Description |
|--------|--------|---------|-------------|
| `Date` | `Mon DD, YYYY` or `Mon DD` | `Jan 27, 2025` | Daily date |
| `productivity percentage` | 0-100 | `75` | Self-rated productivity |
| `sleep duration` | `Xh Ym` or `X:Y` | `7h38m` | Total sleep |

### Recommended Columns

These add more insight to your trait profile:

| Column | Format | Example | Description |
|--------|--------|---------|-------------|
| `Reflection` | Free text | `Had a productive morning...` | Daily reflection |
| `primary mood` | Text | `happy`, `tired`, `neutral` | How you felt |
| `secondary mood` | Text | `energetic`, `anxious` | Secondary emotion |
| `when most productive` | Code 1-5 | `1` = morning, `2` = afternoon | Peak productivity time |
| `studied at home` | `yes`/`no` | `yes` | Worked from home |
| `studied at school` | `yes`/`no` | `no` | Worked at office/school |
| `workout did` | `yes`/`no` | `yes` | Exercised |
| `meditation` | `yes`/`no` | `no` | Meditated |
| `morning shower` | `yes`/`no` | `yes` | Morning routine |
| `played sports` | `yes`/`no` | `no` | Sports activity |
| `sickness` | `yes`/`no` | `no` | Felt unwell |
| `nap today` | `yes`/`no` | `no` | Took a nap |

### Social Interactions

Track how interactions went:

| Column | Values | Description |
|--------|--------|-------------|
| `interaction w/ partner` | `positive`/`neutral`/`negative`/`na` | Partner interaction quality |
| `interaction w/ family` | `positive`/`neutral`/`negative`/`na` | Family interaction quality |
| `interaction w/ friends` | `positive`/`neutral`/`negative`/`na` | Friend interaction quality |

Use `na` when you had no interaction that day.

### Time Columns

| Column | Format | Example | Description |
|--------|--------|---------|-------------|
| `wakeup time` | `H:MM AM/PM` | `6:30 AM` | When you woke up |
| `bed time` | `H:MM AM/PM` | `10:30 PM` | When you went to bed |
| `dinner time` | `H:MM AM/PM` | `7:00 PM` | When you had dinner |

### Sleep Quality

| Column | Format | Example | Description |
|--------|--------|---------|-------------|
| `deep sleep percentage` | 0-100 | `22` | Deep sleep % |
| `REM sleep percentage` | 0-100 | `18` | REM sleep % |

### Nutrition (Optional)

| Column | Values | Description |
|--------|--------|-------------|
| `breakfast quality` | `balanced`/`carb_heavy`/`protein_heavy`/`fat_heavy`/`na` | Breakfast type |
| `lunch quality` | Same as above | Lunch type |
| `dinner quality` | Same as above | Dinner type |
| `water drank` | Number (liters) | `2.5` | Water intake |

---

## Productivity Codes

The `when most productive` field uses these codes:

| Code | Meaning |
|------|---------|
| `1` | Morning |
| `2` | Afternoon |
| `3` | Evening |
| `12` | Morning + Afternoon |
| `13` | Morning + Evening |
| `23` | Afternoon + Evening |
| `123` | All day |
| `5` | Not productive |

---

## Sample CSV

```csv
Date,Reflection,primary mood,productivity percentage,when most productive,sleep duration,wakeup time,bed time,studied at home,workout did
"Jan 27, 2025","Great focus today, finished the report",happy,85,1,7h30m,6:30 AM,10:30 PM,yes,yes
"Jan 28, 2025","Tired from yesterday, took it easy",tired,40,3,6h15m,7:00 AM,11:45 PM,no,no
"Jan 29, 2025","Back on track, good team meeting",productive,75,12,7h45m,6:15 AM,10:15 PM,yes,yes
```

---

## Data Collection Tips

### Start Simple
Begin with just 3-5 columns. Add more as it becomes habit.

**Minimum viable tracking:**
- Date
- Productivity (0-100)
- Sleep duration
- One sentence reflection

### Be Consistent
- Track at the same time each day (e.g., before bed)
- Use reminders or habits to maintain consistency
- 30 days minimum for meaningful patterns

### Be Honest
- Rate productivity objectively, not aspirationally
- Include bad days — they're valuable data
- Reflections don't need to be long

### Use Templates

**Google Sheets Template:**
1. Create a new spreadsheet
2. Add column headers from the schema above
3. Fill one row per day
4. Export → Download as CSV

**Notion Template:**
1. Create a database with properties matching the schema
2. Add a daily entry
3. Export to CSV when ready

---

## Integrating with Wearables

### Apple Health / Health Connect
- Export sleep data
- Map to `sleep duration`, `deep sleep percentage`, `REM sleep percentage`

### Oura Ring
- Export daily readiness and sleep data
- Map sleep stages to percentages

### Fitbit / Garmin
- Export sleep logs
- Include in your CSV

### Manual Integration
Most wearables export CSVs. You can:
1. Export wearable data
2. Merge with your manual tracking spreadsheet
3. Run through CTE

---

## Processing Your Data

Once you have a CSV:

```bash
# Clean the data
poetry run python src/cte/data.py \
  --in your_data.csv \
  --out data/interim/clean.parquet

# Run feature engineering
poetry run python src/cte/features.py \
  --in data/interim/clean.parquet \
  --out data/processed/features.parquet

# Generate a persona (use the notebook or create a script)
```

---

## Troubleshooting

### "Column not found" errors
- Check that your column names match the expected format
- The cleaning pipeline handles common variations (typos, newlines)

### Date parsing issues
- Use consistent date format: `Mon DD, YYYY` or `Mon DD`
- Avoid numeric-only dates like `01/27/25`

### Missing values
- Leave cells empty (not "N/A" or "-")
- The pipeline handles missing values gracefully

---

## Privacy Notes

- All processing happens locally on your machine
- Raw data in `data/raw/` is gitignored by default
- Never commit personal data to version control
- You can use synthetic data for demos instead

---

## Questions?

Open an issue on GitHub or check the main README for more details.
