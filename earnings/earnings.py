# send daily Email on companies that are schedule to report thier earnings on the next day
#!/usr/bin/env python3

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='requests')

import os
import requests
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Email configuration from environment variables
EMAIL_CONFIG = {
    # Gmail SMTP (if you want to try SMTP again)
    'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
    'smtp_port': int(os.getenv('SMTP_PORT', '587')),
    'sender_email': os.getenv('SENDER_EMAIL'),
    'sender_password': os.getenv('SENDER_PASSWORD'),

    # SendGrid configuration
    'sendgrid_api_key': os.getenv('SENDGRID_API_KEY'),

    # Recipients (comma-separated in .env)
    'recipients': os.getenv('RECIPIENTS', '').split(',') if os.getenv('RECIPIENTS') else [],

    # Email service to use: 'sendgrid', 'smtp', or 'file'
    'email_service': os.getenv('EMAIL_SERVICE', 'sendgrid')
}

def get_nasdaq_earnings():
    """Get company earnings from NASDAQ"""
    print("📊 Fetching earnings from NASDAQ...")

    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    api_url = f"https://api.nasdaq.com/api/calendar/earnings?date={tomorrow}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json, text/plain, */*'
    }

    try:
        response = requests.get(api_url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()

            earnings_data = []
            if 'data' in data and 'rows' in data['data']:
                for row in data['data']['rows']:
                    eps_value = row.get('epsForecast', '')
                    eps_parsed = None
                    if eps_value and eps_value != '':
                        try:
                            eps_clean = eps_value.replace('$', '').replace('(', '-').replace(')', '')
                            eps_parsed = float(eps_clean) if eps_clean else None
                        except:
                            eps_parsed = None

                    earnings_data.append({
                        'symbol': row.get('symbol', 'N/A'),
                        'company': row.get('name', 'N/A'),
                        'time': row.get('time', 'time-not-supplied'),
                        'eps': eps_parsed,
                        'eps_raw': eps_value
                    })

            return earnings_data

    except Exception as e:
        print(f"❌ NASDAQ error: {e}")
        return []

def generate_html_report(earnings_data):
    """Generate HTML email report"""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%A, %B %d, %Y')

    # Calculate statistics
    total_companies = len(earnings_data)
    profitable = len([e for e in earnings_data if e['eps'] and e['eps'] > 0])
    losses = len([e for e in earnings_data if e['eps'] and e['eps'] < 0])
    unknown = len([e for e in earnings_data if e['eps'] is None])
    pre_market = len([e for e in earnings_data if e['time'] == 'time-pre-market'])
    after_hours = len([e for e in earnings_data if e['time'] == 'time-after-hours'])

    # Find top performer
    profitable_companies = [e for e in earnings_data if e['eps'] and e['eps'] > 0]
    top_performer = max(profitable_companies, key=lambda x: x['eps']) if profitable_companies else None

    # Generate company rows
    company_rows = ""
    for company in earnings_data:
        eps_display = f"${company['eps']:.2f}" if company['eps'] is not None else "N/A"
        eps_color = "#28a745" if (company['eps'] and company['eps'] > 0) else "#dc3545" if (company['eps'] and company['eps'] < 0) else "#6c757d"

        time_display = {
            'time-pre-market': 'Pre-Market',
            'time-after-hours': 'After Hours',
            'time-not-supplied': 'TBD'
        }.get(company['time'], 'TBD')

        time_color = {
            'time-pre-market': '#007bff',
            'time-after-hours': '#6f42c1',
            'time-not-supplied': '#6c757d'
        }.get(company['time'], '#6c757d')

        company_rows += f"""
        <tr style="border-bottom: 1px solid #e9ecef;">
            <td style="padding: 12px; font-weight: bold; font-size: 16px;">{company['symbol']}</td>
            <td style="padding: 12px; color: #666;">{company['company'][:40]}...</td>
            <td style="padding: 12px;">
                <span style="background-color: {time_color}20; color: {time_color}; padding: 4px 8px; border-radius: 12px; font-size: 12px; font-weight: bold;">
                    {time_display}
                </span>
            </td>
            <td style="padding: 12px; font-weight: bold; color: {eps_color}; font-size: 16px;">{eps_display}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Earnings Report</title>
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; background-color: #f8f9fa;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; color: white;">
            <h1 style="margin: 0; font-size: 36px; font-weight: bold;">📊 EARNINGS CALENDAR</h1>
            <p style="margin: 10px 0 0 0; font-size: 18px; opacity: 0.9;">{tomorrow}</p>
            <div style="background: rgba(255,255,255,0.2); padding: 10px 20px; border-radius: 20px; display: inline-block; margin-top: 15px;">
                <span style="font-weight: bold; font-size: 18px;">🏢 {total_companies} Companies Reporting</span>
            </div>
        </div>

        <div style="padding: 30px; background: white;">
            <!-- Statistics -->
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 20px; margin-bottom: 30px;">
                <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; text-align: center; border-left: 5px solid #28a745;">
                    <div style="font-size: 28px; font-weight: bold; color: #28a745;">{profitable}</div>
                    <div style="font-size: 12px; color: #666; text-transform: uppercase;">Profitable</div>
                </div>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; text-align: center; border-left: 5px solid #dc3545;">
                    <div style="font-size: 28px; font-weight: bold; color: #dc3545;">{losses}</div>
                    <div style="font-size: 12px; color: #666; text-transform: uppercase;">Losses</div>
                </div>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; text-align: center; border-left: 5px solid #007bff;">
                    <div style="font-size: 28px; font-weight: bold; color: #007bff;">{pre_market}</div>
                    <div style="font-size: 12px; color: #666; text-transform: uppercase;">Pre-Market</div>
                </div>
                <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; text-align: center; border-left: 5px solid #6f42c1;">
                    <div style="font-size: 28px; font-weight: bold; color: #6f42c1;">{after_hours}</div>
                    <div style="font-size: 12px; color: #666; text-transform: uppercase;">After Hours</div>
                </div>
            </div>

            {f'''
            <!-- Top Performer -->
            <div style="background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 30px;">
                <div style="font-size: 18px; font-weight: bold; margin-bottom: 10px;">🏆 TOP PERFORMER</div>
                <div style="font-size: 32px; font-weight: bold;">{top_performer['symbol']} • ${top_performer['eps']:.2f}</div>
                <div style="font-size: 14px; opacity: 0.9;">{top_performer['company'][:50]}</div>
            </div>
            ''' if top_performer else ''}

            <!-- Companies Table -->
            <div style="background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 20px;">
                    <h2 style="margin: 0; font-size: 24px; font-weight: bold;">📈 Companies Reporting Tomorrow</h2>
                </div>
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 15px; text-align: left; font-weight: bold; color: #333;">Symbol</th>
                            <th style="padding: 15px; text-align: left; font-weight: bold; color: #333;">Company</th>
                            <th style="padding: 15px; text-align: left; font-weight: bold; color: #333;">Timing</th>
                            <th style="padding: 15px; text-align: left; font-weight: bold; color: #333;">EPS Est.</th>
                        </tr>
                    </thead>
                    <tbody>
                        {company_rows}
                    </tbody>
                </table>
            </div>

            <!-- Footer -->
            <div style="margin-top: 30px; padding-top: 20px; border-top: 2px solid #e9ecef; text-align: center; color: #666; font-size: 12px;">
                <strong>📊 Data Source: NASDAQ</strong> • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                EPS = Earnings Per Share (Analyst Estimates) • Pre-Market: Before 9:30 AM • After Hours: After 4:00 PM
            </div>
        </div>
    </body>
    </html>
    """

    return html_content

def send_email_sendgrid(subject, html_content, recipients):
    """Send email using SendGrid API"""

    api_key = EMAIL_CONFIG['sendgrid_api_key']
    if not api_key:
        print("❌ SendGrid API key not found in environment variables")
        return False

    url = "https://api.sendgrid.com/v3/mail/send"

    email_data = {
        "personalizations": [
            {
                "to": [{"email": email} for email in recipients],
                "subject": subject
            }
        ],
        "from": {
            "email": EMAIL_CONFIG['sender_email'],
            "name": "Earnings Alert System"
        },
        "content": [
            {
                "type": "text/html",
                "value": html_content
            }
        ]
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, json=email_data)

        if response.status_code == 202:
            print(f"✅ Email sent successfully via SendGrid to {len(recipients)} recipients")
            return True
        else:
            print(f"❌ SendGrid error: {response.status_code} - {response.text}")
            return False

    except Exception as e:
        print(f"❌ Failed to send email via SendGrid: {e}")
        return False

def send_email_smtp(subject, html_content, recipients):
    """Send email using SMTP (fallback option)"""
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    if not EMAIL_CONFIG['sender_password']:
        print("❌ SMTP password not found in environment variables")
        return False

    try:
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = EMAIL_CONFIG['sender_email']
        msg['To'] = ', '.join(recipients)

        html_part = MIMEText(html_content, 'html')
        msg.attach(html_part)

        # Try SSL first, then TLS
        try:
            print("🔐 Trying SSL connection...")
            server = smtplib.SMTP_SSL(EMAIL_CONFIG['smtp_server'], 465)
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
        except Exception:
            print("🔐 Trying TLS connection...")
            server = smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'])
            server.starttls()
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])

        server.sendmail(EMAIL_CONFIG['sender_email'], recipients, msg.as_string())
        server.quit()

        print(f"✅ Email sent successfully via SMTP to {len(recipients)} recipients")
        return True

    except Exception as e:
        print(f"❌ SMTP failed: {e}")
        return False

def save_to_file(subject, html_content):
    """Save report to file as fallback"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"earnings_report_{timestamp}.html"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"✅ Report saved to file: {filename}")
        print(f"💡 Open {filename} in your browser to view the report")
        return True

    except Exception as e:
        print(f"❌ Failed to save file: {e}")
        return False

def validate_config():
    """Validate environment configuration"""
    errors = []

    if not EMAIL_CONFIG['sender_email']:
        errors.append("SENDER_EMAIL is required")

    if not EMAIL_CONFIG['recipients'] or EMAIL_CONFIG['recipients'] == ['']:
        errors.append("RECIPIENTS is required")

    service = EMAIL_CONFIG['email_service']
    if service == 'sendgrid' and not EMAIL_CONFIG['sendgrid_api_key']:
        errors.append("SENDGRID_API_KEY is required when using SendGrid service")

    if service == 'smtp' and not EMAIL_CONFIG['sender_password']:
        errors.append("SENDER_PASSWORD is required when using SMTP service")

    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   • {error}")
        print("\n💡 Check your .env file and make sure all required variables are set")
        return False

    return True

def main():
    """Main function for cron job"""
    print("🔍 Starting earnings email system...")

    # Validate configuration
    if not validate_config():
        return

    tomorrow_date = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    print(f"📅 Fetching earnings for {tomorrow_date}")

    # Get earnings data
    earnings_data = get_nasdaq_earnings()

    if not earnings_data:
        print("❌ No earnings data found - not sending email")
        return

    print(f"✅ Found {len(earnings_data)} companies with earnings")

    # Generate HTML report
    html_report = generate_html_report(earnings_data)
    subject = f"📊 Earnings Calendar: {len(earnings_data)} Companies Reporting {tomorrow_date}"

    # Send email based on configured service
    service = EMAIL_CONFIG['email_service']
    success = False

    if service == 'sendgrid':
        print("📧 Sending via SendGrid...")
        success = send_email_sendgrid(subject, html_report, EMAIL_CONFIG['recipients'])
    elif service == 'smtp':
        print("📧 Sending via SMTP...")
        success = send_email_smtp(subject, html_report, EMAIL_CONFIG['recipients'])
    else:
        print("💾 Saving to file...")
        success = save_to_file(subject, html_report)

    # Fallback to file if email fails
    if not success and service != 'file':
        print("📄 Email failed, saving to file as fallback...")
        save_to_file(subject, html_report)

if __name__ == "__main__":
    main()
