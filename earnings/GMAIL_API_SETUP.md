# Gmail Cloud API Setup Guide

This guide explains how to set up the Gmail Cloud API for sending emails when SMTP ports (465, 587) are blocked by firewalls.

## Why Gmail Cloud API?

The Gmail Cloud API uses HTTPS (port 443) instead of SMTP ports, making it work in restrictive network environments where SMTP is blocked. The script automatically falls back to the Gmail Cloud API when SMTP fails.

## Setup Steps

### 1. Install Required Dependencies

On the remote computer, install the Gmail Cloud API libraries:

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client
```

### 2. Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select an existing one)
3. Enable the Gmail Cloud API:
   - Go to "APIs & Services" > "Library"
   - Search for "Gmail Cloud API" (or just "Gmail")
   - Click "Enable"

### 3. Create OAuth2 Credentials

1. Go to "APIs & Services" > "Credentials"
2. Click "Create Credentials" > "OAuth client ID"
3. If prompted, configure the OAuth consent screen:
   - Choose "External" user type
   - Fill in the required app information
   - Add your email as a test user
   - Add scope: `https://www.googleapis.com/auth/gmail.send`
4. For "Application type", select "Desktop app"
5. Give it a name (e.g., "Earnings Report Emailer")
6. Click "Create"
7. Download the credentials JSON file

### 4. Set Up Credentials File

1. Rename the downloaded file to `credentials.json`
2. Copy it to the same directory as `earnings.py`:

```bash
# On your main computer (where you downloaded credentials.json)
scp credentials.json remote-computer:/path/to/earnings/

# Or if using the same machine, just move it:
mv ~/Downloads/credentials.json /path/to/earnings/
```

### 5. First-Time Authentication

The first time you run the script, it will:
1. Open a browser window for authentication
2. Ask you to log in to your Google account
3. Request permission to send emails
4. Save the authentication token to `token.pickle`

**Important**: This step requires a browser and interactive session. You have two options:

#### Option A: Authenticate locally first (Recommended)
```bash
# On a computer with a browser, run the script once:
python3 earnings.py

# This creates token.pickle
# Then copy token.pickle to the remote computer:
scp token.pickle remote-computer:/path/to/earnings/
```

#### Option B: SSH with X11 forwarding
```bash
# SSH to remote computer with X11 forwarding
ssh -X remote-computer

# Run the script (browser will forward to your local machine)
cd /path/to/earnings
python3 earnings.py
```

### 6. Verify Setup

After setup, your directory should contain:
- `earnings.py` (the script)
- `credentials.json` (OAuth2 credentials)
- `token.pickle` (created after first auth)
- `.env` (your email configuration)

## How the Fallback System Works

The script tries email methods in this order:

1. **Gmail SMTP (port 465)** - Fastest if available
2. **Gmail SMTP (port 587)** - Alternative SMTP port
3. **Gmail Cloud API (HTTPS)** - Uses port 443, works when SMTP is blocked
4. **Save to file** - Last resort fallback

You'll see output like:
```
📧 Attempting to send email via Gmail SMTP...
⚠️  Failed to send via SMTP port 465 (SMTP_SSL): [Errno 110] Connection timed out
🔄 Trying next port...
⚠️  Failed to send via SMTP port 587 (STARTTLS): [Errno 110] Connection timed out
❌ All SMTP ports failed
🔄 SMTP failed, trying Gmail Cloud API...
✅ Email sent successfully via Gmail Cloud API to 3 recipients
```

## Security Notes

- `credentials.json` contains your OAuth2 client credentials (not your password)
- `token.pickle` contains your access/refresh tokens
- Both files should be kept secure and not committed to version control
- The `.env` file with `SENDER_PASSWORD` is only needed for SMTP (can be empty if using API only)

## Troubleshooting

### "credentials.json not found"
- Make sure you downloaded the OAuth2 credentials and renamed it to `credentials.json`
- Place it in the same directory as `earnings.py`

### "ImportError: No module named google"
- Install the required packages: `pip install google-auth-oauthlib google-auth-httplib2 google-api-python-client`

### Authentication browser doesn't open
- Use Option A above: authenticate on a computer with a browser, then copy `token.pickle`
- Or use SSH with X11 forwarding

### "Token has been expired or revoked"
- Delete `token.pickle` and run the authentication flow again

### Still can't connect
- Check if port 443 (HTTPS) is blocked: `curl -I https://www.googleapis.com`
- If even HTTPS is blocked, you may need to use a VPN or SSH tunnel

## Maintenance

The `token.pickle` file contains a refresh token that automatically renews your access. You typically don't need to re-authenticate unless:
- The token is revoked
- You change Google accounts
- The OAuth2 consent screen expires (after 7 days for unverified apps with external users)

If emails stop working, delete `token.pickle` and re-authenticate.
