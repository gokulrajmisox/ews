
with open('d:/ews/backend/api/router.py', 'a', encoding='utf-8') as f:
    f.write('''

import requests

@router.post("/telegram_alert")
async def send_telegram_alert(patient_id: int):
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not token or not chat_id:
        return {"success": False, "message": "Telegram credentials not configured in environment."}
        
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    text = f"dY"" CRITICAL ALERT: Patient {patient_id} has entered the ALERT state! Immediate review recommended."
    
    try:
        resp = requests.post(url, json={"chat_id": chat_id, "text": text})
        if resp.status_code == 200:
            return {"success": True, "message": "Alert sent via Telegram."}
        else:
            return {"success": False, "message": f"Telegram API error: {resp.text}"}
    except Exception as e:
        return {"success": False, "message": str(e)}
''')
