
import json
import logging

with open('d:/ews/backend/api/router.py', 'a', encoding='utf-8') as f:
    f.write('''

from fastapi import UploadFile, File, Form
from google import genai
from google.genai import types
from google.genai.errors import APIError

@router.post("/chat")
async def chat_with_ai(
    message: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        model = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")
        if not api_key:
            return {"reply": "Error: GEMINI_API_KEY is not set in the environment."}
            
        client = genai.Client(api_key=api_key)
        
        prompt = ""
        if file:
            content = await file.read()
            csv_text = content.decode('utf-8')
            prompt += f"Here is the patient CSV data:\\n{csv_text}\\n\\n"
            
        if message:
            prompt += message
        else:
            prompt += "Please analyze this patient data."
            
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return {"reply": response.text}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"reply": f"An error occurred: {str(e)}"}
''')
