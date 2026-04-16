# TSETMC — داشبورد بازار بورس ایران

## Quick Reference

- **Framework:** Django 6.x (Python 3.12)
- **Database:** SQLite (dev), PostgreSQL-ready
- **Language/Direction:** Persian (fa-ir), RTL
- **Font:** Vazir (loaded from `/static/core/fonts/vazir/`)
- **UI Theme:** Dark glassmorphism (CSS vars in `:root`, no CSS framework)
- **Deploy:** Docker + Gunicorn + WhiteNoise

## Commands

```bash
python manage.py runserver          # dev server
python manage.py makemigrations     # after model changes
python manage.py migrate            # apply migrations
python manage.py createsuperuser    # admin user
python manage.py collectstatic      # for production
```

## Project Structure

```
config/           → Django settings, urls, wsgi
core/             → Main app (market dashboard, symbol analysis)
  service/        → Business logic (analyzer.py = technical analysis engine)
  data_providers/ → External API wrappers (brsapi, finpy)
  templates/core/ → home.html, symbol.html, no_subscription.html
  static/core/    → Fonts (Vazir)
accounts/         → Auth app (login, register, profile, subscription)
  templates/accounts/ → login.html, register.html, account.html
templates/admin/  → Django admin template overrides (RTL font fix)
```

## Architecture

### Data Flow
- **Live market data** comes from BrsApi.ir (API key in `BRS_API_KEY` env var)
- **Historical prices** come from `finpy-tse` library (TSETMC scraper)
- **Caching:** Django LocMemCache with time-based TTLs (shorter during market hours 09:01-18:10 Tehran)

### Authentication & Subscription
- Uses Django's built-in `User` model + custom `UserProfile` (OneToOne)
- `UserProfile.days_remaining`: daily credit system, decrements once per day on first access
- Symbol analysis (`/symbol/` page + `/api/analyze/` API) requires login + active subscription
- Dashboard (`/`) is public, no auth required
- Admin panel at `/admin/` manages users and subscription credits

### Key APIs
```
GET /                           → Dashboard (public)
GET /symbol/                    → Analysis page (auth + subscription)
GET /api/analyze/?symbol=X      → Technical analysis JSON (auth + subscription)
GET /api/tsetmc/index/?type=N   → Market indices (public)
GET /api/market/gold-currency/  → Gold/currency prices (public)
GET /api/tsetmc/symbol-live/?symbol=X → Live symbol data (public)
```

## Conventions

- All templates are standalone HTML (no base template inheritance) with inline CSS
- CSS variables defined in `:root` block, consistent across all pages
- Persian text everywhere in UI — use Vazir font, `letter-spacing: 0`
- Django admin has custom RTL fix in `templates/admin/base_site.html`
- API responses use `json_dumps_params={"ensure_ascii": False}` for Persian
- Error responses return JSON with `error` + `detail` fields

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `BRS_API_KEY` | prod only | BrsApi.ir API key |
| `DJANGO_SECRET_KEY` | prod only | Django secret key |
| `DJANGO_DEBUG` | no | "True"/"False" (default: True) |

## Things to Watch Out For

- `letter-spacing` in CSS breaks Persian script — always set to `0`
- `finpy-tse` can fail (network issues) — analyzer gracefully degrades to live-only analysis
- Admin `UserProfile` inline + `get_or_create` pattern — don't create profile in both `save_model` and inline
- Templates have no shared base — changes to navbar must be applied to each template individually
