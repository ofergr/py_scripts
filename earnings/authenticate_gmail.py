#!/usr/bin/env python3
"""
Helper script to authenticate Gmail API in headless/remote environments.
Run this on a machine with a browser, then copy token.pickle to your server.
"""

import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ['https://www.googleapis.com/auth/gmail.send']

def authenticate():
    """Run OAuth flow and save credentials"""
    if not os.path.exists('credentials.json'):
        print("❌ Error: credentials.json not found!")
        print("\nTo get credentials.json:")
        print("1. Go to https://console.cloud.google.com/")
        print("2. Create a project or select existing one")
        print("3. Enable Gmail API")
        print("4. Go to 'Credentials' > 'Create Credentials' > 'OAuth 2.0 Client ID'")
        print("5. Choose 'Desktop app' as application type")
        print("6. Download the JSON file and save as 'credentials.json'")
        return False

    print("🔐 Starting Gmail OAuth authentication...")
    print("⏳ A browser window will open. Please authorize the application.")

    try:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        # Use port=0 to let the OS choose an available port automatically
        creds = flow.run_local_server(port=0)

        # Save the credentials
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

        print("\n✅ Authentication successful!")
        print("📁 token.pickle has been created")
        print("\nIf you're authenticating on a different machine:")
        print("  1. Copy token.pickle to your server")
        print("  2. Place it in the same directory as earnings.py")
        print("  3. Run earnings.py again")

        return True

    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        return False

if __name__ == "__main__":
    authenticate()
