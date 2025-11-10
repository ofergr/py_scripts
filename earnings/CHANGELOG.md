# Changelog

## Version 2.3 - Ticker Logo

### New Features

#### Adding Companies Logo To Report
- Add the company logo near its ticker name for easy indetification

---

## Version 2.2 - Logging Implementation

### New Features

#### Comprehensive Logging System
- **Replaced all print statements with structured logging**
- Daily rotating log files: `logs/earnings_YYYYMMDD.log`
- Log levels: ERROR (❌), WARNING (⚠️), INFO (✅📊📋📧), DEBUG (📅)
- Dual output: Both file and console for flexibility
- Perfect for cron job monitoring and debugging

### Files Modified
- `earnings.py` - All 57 print statements converted to logger calls
- `.gitignore` - Added `logs/` and `*.log`
- `GMAIL_API_SETUP.md` - Added logging documentation section
- `CHANGELOG.md` - This file

### Benefits
- Persistent logs survive cron job execution
- Easy error tracking: `grep ERROR logs/*.log`
- Historical debugging with daily log files
- No loss of diagnostic information

---

## Version 2.1 - Gmail API Integration

### New Features

#### Gmail API Support
- **Multi-tier email fallback system**: SMTP (465) → SMTP (587) → Gmail API → File save
- Gmail API uses HTTPS (port 443), bypassing firewall restrictions on SMTP ports
- Automatic fallback ensures email delivery in restrictive network environments
- Python 3.9+ compatibility shim for `importlib.metadata`

#### Comprehensive Logging
- Replaced all `print()` statements with structured logging
- Daily rotating log files in `logs/` directory
- Log levels: ERROR, WARNING, INFO, DEBUG
- Both file and console output for flexibility
- Perfect for cron job monitoring

### Technical Improvements

#### Dependencies Added
- `google-auth-oauthlib>=1.0.0` - OAuth2 authentication
- `google-auth-httplib2>=0.1.0` - HTTP library for Google APIs
- `google-api-python-client>=2.0.0` - Gmail API client

#### Security
- Added `credentials.json` and `token.pickle` to `.gitignore`
- OAuth2 token-based authentication (no password storage for API)
- Automatic token refresh

#### Files Added
- `GMAIL_API_SETUP.md` - Complete setup guide for Gmail API
- `authenticate_gmail.py` - Standalone authentication tool
- `CHANGELOG.md` - This file
- `logs/` directory - Daily log files

### Setup Required

For systems with blocked SMTP ports:

1. Install new dependencies: `pip install -r requirements.txt`
2. Set up Google Cloud OAuth2 credentials (see `GMAIL_API_SETUP.md`)
3. Run authentication: `python3 authenticate_gmail.py`
4. Copy `credentials.json` and `token.pickle` to production server

### Backward Compatibility

- Existing SMTP configuration still works
- No changes required if SMTP ports are accessible
- Gmail API is transparent fallback

### Bug Fixes

- Fixed Python 3.9 compatibility with Google API libraries
- Added proper timeout handling for SMTP connections (10s)
- Improved error messages for troubleshooting

---

## Previous Versions

See git history for changes prior to v2.1
