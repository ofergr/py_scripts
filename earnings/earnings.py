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
    'sendgrid_api_key': os.getenv('SENDGRID_API_KEY'),
    'sender_email': os.getenv('SENDER_EMAIL'),
    'recipients': os.getenv('RECIPIENTS', '').split(',') if os.getenv('RECIPIENTS') else [],
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
    """Generate clean HTML email report using only NASDAQ data"""
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%A, %B %d, %Y')
    
    # Calculate statistics
    total_companies = len(earnings_data)
    profitable = len([e for e in earnings_data if e.get('eps') and e.get('eps') > 0])
    losses = len([e for e in earnings_data if e.get('eps') and e.get('eps') < 0])
    unknown = len([e for e in earnings_data if e.get('eps') is None])
    pre_market = len([e for e in earnings_data if e.get('time') == 'time-pre-market'])
    after_hours = len([e for e in earnings_data if e.get('time') == 'time-after-hours'])
    
    # Find top performer
    profitable_companies = [e for e in earnings_data if e.get('eps') and e.get('eps') > 0]
    top_performer = max(profitable_companies, key=lambda x: x.get('eps', 0)) if profitable_companies else None
    
    # Find biggest loss
    loss_companies = [e for e in earnings_data if e.get('eps') and e.get('eps') < 0]
    biggest_loss = min(loss_companies, key=lambda x: x.get('eps', 0)) if loss_companies else None
    
    # Generate company rows
    company_rows = ""
    for company in earnings_data:
        symbol = company.get('symbol', 'N/A')
        company_name = company.get('company', 'N/A')
        time_value = company.get('time', 'time-not-supplied')
        eps_value = company.get('eps')
        
        # Format EPS display
        eps_display = f"${eps_value:.2f}" if eps_value is not None else "N/A"
        eps_color = "#28a745" if (eps_value and eps_value > 0) else "#dc3545" if (eps_value and eps_value < 0) else "#6c757d"
        
        # Format timing
        time_display = {
            'time-pre-market': 'Pre-Market',
            'time-after-hours': 'After Hours',
            'time-not-supplied': 'TBD'
        }.get(time_value, 'TBD')
        
        time_color = {
            'time-pre-market': '#007bff',
            'time-after-hours': '#6f42c1',
            'time-not-supplied': '#6c757d'
        }.get(time_value, '#6c757d')
        
        # Add indicator for significant EPS
        eps_indicator = ""
        if eps_value:
            if eps_value >= 2.0:
                eps_indicator = "🔥"  # Hot stock
            elif eps_value >= 1.0:
                eps_indicator = "📈"  # Good performance
            elif eps_value > 0:
                eps_indicator = "✅"  # Profitable
            else:
                eps_indicator = "⚠️"   # Loss
        
        company_rows += f"""
        <tr style="border-bottom: 1px solid #e9ecef;">
            <td style="padding: 15px; font-weight: bold; font-size: 18px;">{symbol}</td>
            <td style="padding: 15px;">
                <div style="color: #333; font-weight: 500; font-size: 16px;">{company_name[:45]}...</div>
            </td>
            <td style="padding: 15px; text-align: center;">
                <span style="background-color: {time_color}20; color: {time_color}; padding: 6px 12px; border-radius: 15px; font-size: 13px; font-weight: bold;">
                    {time_display}
                </span>
            </td>
            <td style="padding: 15px; text-align: center;">
                <div style="display: flex; align-items: center; justify-content: center; gap: 8px;">
                    <span style="font-size: 20px;">{eps_indicator}</span>
                    <span style="font-weight: bold; color: {eps_color}; font-size: 18px;">{eps_display}</span>
                </div>
            </td>
        </tr>
        """
    
    # Generate highlights section
    highlights_section = ""
    if top_performer or biggest_loss:
        highlights_section = f"""
        <!-- Highlights -->
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px;">
            {f'''
            <div style="background: linear-gradient(135deg, #28a745, #20c997); color: white; padding: 25px; border-radius: 15px; text-align: center;">
                <div style="font-size: 16px; font-weight: bold; margin-bottom: 10px;">🏆 HIGHEST EPS</div>
                <div style="font-size: 28px; font-weight: bold;">{top_performer['symbol']} • ${top_performer['eps']:.2f}</div>
                <div style="font-size: 14px; opacity: 0.9; margin-top: 5px;">{top_performer['company'][:35]}...</div>
            </div>
            ''' if top_performer else ''}
            
            {f'''
            <div style="background: linear-gradient(135deg, #dc3545, #c82333); color: white; padding: 25px; border-radius: 15px; text-align: center;">
                <div style="font-size: 16px; font-weight: bold; margin-bottom: 10px;">📉 BIGGEST LOSS</div>
                <div style="font-size: 28px; font-weight: bold;">{biggest_loss['symbol']} • ${biggest_loss['eps']:.2f}</div>
                <div style="font-size: 14px; opacity: 0.9; margin-top: 5px;">{biggest_loss['company'][:35]}...</div>
            </div>
            ''' if biggest_loss else ''}
        </div>
        """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Earnings Report</title>
    </head>
    <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 900px; margin: 0 auto; background-color: #f8f9fa;">
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; text-align: center; color: white;">
            <h1 style="margin: 0; font-size: 42px; font-weight: bold;">📊 EARNINGS CALENDAR</h1>
            <p style="margin: 15px 0 0 0; font-size: 20px; opacity: 0.9;">{tomorrow}</p>
            <div style="background: rgba(255,255,255,0.2); padding: 15px 25px; border-radius: 25px; display: inline-block; margin-top: 20px;">
                <span style="font-weight: bold; font-size: 20px;">🏢 {total_companies} Companies Reporting</span>
            </div>
        </div>
        
        <div style="padding: 40px; background: white;">
            <!-- Statistics -->
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 25px; margin-bottom: 40px;">
                <div style="background: #f8f9fa; padding: 25px; border-radius: 15px; text-align: center; border-left: 6px solid #28a745; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <div style="font-size: 32px; font-weight: bold; color: #28a745; margin-bottom: 5px;">📈 {profitable}</div>
                    <div style="font-size: 14px; color: #666; text-transform: uppercase; font-weight: 600;">Profitable</div>
                </div>
                <div style="background: #f8f9fa; padding: 25px; border-radius: 15px; text-align: center; border-left: 6px solid #dc3545; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <div style="font-size: 32px; font-weight: bold; color: #dc3545; margin-bottom: 5px;">📉 {losses}</div>
                    <div style="font-size: 14px; color: #666; text-transform: uppercase; font-weight: 600;">Losses</div>
                </div>
                <div style="background: #f8f9fa; padding: 25px; border-radius: 15px; text-align: center; border-left: 6px solid #007bff; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <div style="font-size: 32px; font-weight: bold; color: #007bff; margin-bottom: 5px;">🌅 {pre_market}</div>
                    <div style="font-size: 14px; color: #666; text-transform: uppercase; font-weight: 600;">Pre-Market</div>
                </div>
                <div style="background: #f8f9fa; padding: 25px; border-radius: 15px; text-align: center; border-left: 6px solid #6f42c1; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                    <div style="font-size: 32px; font-weight: bold; color: #6f42c1; margin-bottom: 5px;">🌙 {after_hours}</div>
                    <div style="font-size: 14px; color: #666; text-transform: uppercase; font-weight: 600;">After Hours</div>
                </div>
            </div>
            
            {highlights_section}
            
            <!-- Companies Table -->
            <div style="background: white; border-radius: 15px; overflow: hidden; box-shadow: 0 8px 25px rgba(0,0,0,0.1); border: 1px solid #e9ecef;">
                <div style="background: linear-gradient(135deg, #667eea, #764ba2); color: white; padding: 25px;">
                    <h2 style="margin: 0; font-size: 28px; font-weight: bold;">📈 Companies Reporting Tomorrow</h2>
                </div>
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background: #f8f9fa;">
                            <th style="padding: 20px; text-align: left; font-weight: bold; color: #333; font-size: 16px;">Symbol</th>
                            <th style="padding: 20px; text-align: left; font-weight: bold; color: #333; font-size: 16px;">Company</th>
                            <th style="padding: 20px; text-align: center; font-weight: bold; color: #333; font-size: 16px;">Timing</th>
                            <th style="padding: 20px; text-align: center; font-weight: bold; color: #333; font-size: 16px;">EPS Estimate</th>
                        </tr>
                    </thead>
                    <tbody>
                        {company_rows}
                    </tbody>
                </table>
            </div>
            
            <!-- Legend -->
            <div style="background: #f8f9fa; padding: 20px; border-radius: 12px; margin-top: 30px; border-left: 4px solid #667eea;">
                <h3 style="margin: 0 0 15px 0; color: #333; font-size: 18px;">📋 Legend</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; font-size: 14px;">
                    <div><span style="font-size: 18px;">🔥</span> High Performance (≥$2.00)</div>
                    <div><span style="font-size: 18px;">📈</span> Strong Growth (≥$1.00)</div>
                    <div><span style="font-size: 18px;">✅</span> Profitable (>$0.00)</div>
                    <div><span style="font-size: 18px;">⚠️</span> Expected Loss (<$0.00)</div>
                </div>
            </div>
            
            <!-- Footer -->
            <div style="margin-top: 40px; padding-top: 25px; border-top: 3px solid #e9ecef; text-align: center; color: #666; font-size: 14px;">
                <strong>📊 Data Source: NASDAQ</strong> • Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
                EPS = Earnings Per Share (Analyst Estimates) • Pre-Market: Before 9:30 AM • After Hours: After 4:00 PM<br>
                <br>
                <em>This is an automated financial report. To stop receiving these emails, contact the sender.</em>
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
        "reply_to": {
            "email": EMAIL_CONFIG['sender_email'],
            "name": "Earnings Alert System"
        },
        "content": [
            {
                "type": "text/html",
                "value": html_content
            }
        ],
        "categories": ["earnings-report", "financial-data"]
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
    subject = f"📊 Daily Earnings Report - {len(earnings_data)} Companies - {tomorrow_date}"
    
    # Send email based on configured service
    service = EMAIL_CONFIG['email_service']
    success = False
    
    if service == 'sendgrid':
        print("📧 Sending via SendGrid...")
        success = send_email_sendgrid(subject, html_report, EMAIL_CONFIG['recipients'])
    else:
        print("💾 Saving to file...")
        success = save_to_file(subject, html_report)
    
    # Fallback to file if email fails
    if not success and service != 'file':
        print("📄 Email failed, saving to file as fallback...")
        save_to_file(subject, html_report)

if __name__ == "__main__":
    main()