import os
import json
import asyncio
import re
from typing import Optional, List, Any
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, firestore
import google.generativeai as genai
from google.ai.generativelanguage_v1beta.types import content
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize Firebase Admin SDK
try:
    cred = credentials.Certificate("firebase_credentials.json")
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Firebase: {e}")
    raise

# Configure Gemini API keys
GEMINI_KEY_MAIN = os.getenv("GEMINI_KEY_MAIN")
GEMINI_KEY_CUSTOM = os.getenv("GEMINI_KEY_CUSTOM")
GEMINI_KEY_CLASS = os.getenv("GEMINI_KEY_CLASS")
if not GEMINI_KEY_MAIN or not GEMINI_KEY_CUSTOM or not GEMINI_KEY_CLASS:
    logger.warning("Gemini API keys not found in environment variables")

class ChatRequest(BaseModel):
    userDoc: str  # This should be the userId (parent document)
    agrajaDocId: str  # This should be the agraja document ID (subcollection document)
    prompt: str

async def call_gemini_api(prompt: str, api_key: str) -> str:
    """Call Gemini API with the given prompt and API key"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise HTTPException(status_code=500, detail=f"Gemini API call failed: {str(e)}")

async def classify_and_update(user_id: str, agraja_doc_id: str, full_prompt: str) -> None:
    """Background task to classify user message and update Firestore accordingly"""
    try:
        # Extract user message from the full prompt
        # Look for patterns like "User's Current Message:" or similar
        user_message = ""
        lines = full_prompt.split('\n')
        capture_next = False
        for line in lines:
            if "User's Current Message:" in line or "User Message:" in line:
                capture_next = True
                continue
            if capture_next and line.strip():
                user_message = line.strip()
                break
        
        # If we couldn't extract from prompt structure, use a fallback approach
        if not user_message:
            # Look for the last non-empty line as user message
            for line in reversed(lines):
                if line.strip() and not line.startswith('---') and not line.startswith('**'):
                    user_message = line.strip()
                    break
        # Placeholder for classification prompt - to be filled later
        classification_prompt = f'''
You are a classification and extraction system for a chatbot that speaks to elderly Indian users.  
Your job is to analyze the user's latest message and classify it into one of two special flags, or 0 if neither applies.  
You must also produce a **clear and detailed instruction** for updating the chatbot's memory so that future conversations reflect the user's request or interest.

---

### Flag Definitions

**Flag 1 — Customisation Instruction**  
The user is telling the chatbot how to behave, speak, or address them.  
This is **not** a topic of conversation — this is a permanent setting about the chatbot's style, language, or tone.  
Examples:  
- "Talk to me only in Hindi"  
- "Call me akka, not aunty"  
- "Speak to me in a friendly way"  
- "Don't mention politics when talking to me"  

These are persistent behavioural instructions for the chatbot.

---

**Flag 2 — Personal Interest, Fact, or Memory**  
The user is sharing information about themselves that could be used in future conversations.  
This is **not** a chatbot instruction — it is personal content about the user's life, likes, dislikes, hobbies, past work, or experiences.  
Examples:  
- "I like playing football" (future: can talk about football)  
- "I like the Ramayana" (future: can talk about the Ramayana)  
- "I used to work at the post office" (future: can ask about their experiences there)  
- "I often visit my grandchildren in Delhi" (future: can ask about family visits)  

---

**Flag 0 — Neither**  
The message is general conversation without introducing a permanent customisation or new personal fact.  
Examples:  
- "I am feeling tired today"  
- "What did you do yesterday?"  
- "The weather is nice"  

---

### Output Format
Return the output as a **Dart-style array**:


[flag, "detailed_instruction"]
Where:

flag is an integer: 0, 1, or 2.

detailed_instruction is a clear, self-contained instruction that tells the system exactly what should be stored in the chatbot’s memory.

If flag = 1, the instruction should precisely restate the customisation request so the chatbot can follow it permanently.

If flag = 2, the instruction should clearly describe the personal fact/interest so it can be remembered and used in future conversations.

If flag = 0, the instruction should be an empty string "".

Important Rules for You:

Always choose exactly one flag per message — do not combine them.

For flag 1 or 2, rewrite the user’s message into a clear, future-proof instruction.

Example: If the user says "Call me akka", write "Always address the user as 'akka' instead of 'aunty'."

Example: If the user says "I like playing football", write "The user enjoys playing football; can bring up this topic in the future."

Do not output explanations or reasoning — only return the array.

The instruction must be actionable by another chatbot with no extra context.

User Message
"{user_message}"

Your Output
Return only the Dart-style array as per the above rules.


'''
        
        # Call classification Gemini API
        classification_response = await call_gemini_api(classification_prompt, GEMINI_KEY_CLASS)
        
        # Parse the response to extract flag and instruction
        flag, instruction = parse_gemini_response(classification_response)
        
        # Print flag and instruction for debugging
        print(f"flag={flag}, instruction='{instruction}'")
        
        logger.info(f"Classification result: flag={flag}, instruction='{instruction}' for user {user_id}")
        
        # Handle based on flag
        if flag == 1:  # Customisation instruction
            # Get current customisations from Firestore
            doc_ref = db.collection('users').document(user_id).collection('Agraja').document(agraja_doc_id)
            doc = doc_ref.get()
            current_customisations = ""
            if doc.exists:
                doc_data = doc.to_dict()
                current_customisations = doc_data.get('customisations1', '')
            
            # Update customisations using the parsed instruction
            await update_customisations(user_id, agraja_doc_id, current_customisations, instruction)
            
        elif flag == 2:  # Personal preference or memory
            # Get current interests from Firestore
            doc_ref = db.collection('users').document(user_id).collection('Agraja').document(agraja_doc_id)
            doc = doc_ref.get()
            current_interests = ""
            if doc.exists:
                doc_data = doc.to_dict()
                current_interests = doc_data.get('intrests2', '')
            
            # Update interests using the parsed instruction
            await update_interests(user_id, agraja_doc_id, current_interests, instruction)
            
        # flag == 0: do nothing
        
    except Exception as e:
        logger.error(f"Error in classify_and_update: {e}")

def parse_gemini_response(response_text: str) -> tuple[int, str]:
    """Parse the Gemini response to extract flag and response"""
    try:
        # Remove any markdown formatting
        response_text = response_text.strip()
        if response_text.startswith("```") and response_text.endswith("```"):
            response_text = response_text[3:-3].strip()
        if response_text.startswith("dart"):
            response_text = response_text[4:].strip()
        
        # Find the array pattern
        array_match = re.search(r'\[(\d+),\s*"([^"]+)"\]', response_text)
        if array_match:
            flag = int(array_match.group(1))
            response = array_match.group(2)
            return flag, response
        
        # Fallback: try to extract numbers and quoted strings
        lines = response_text.split('\n')
        for line in lines:
            if '[' in line and ']' in line:
                # Extract content between brackets
                bracket_content = re.search(r'\[(.*?)\]', line)
                if bracket_content:
                    content = bracket_content.group(1)
                    parts = content.split(',', 1)
                    if len(parts) == 2:
                        flag = int(parts[0].strip())
                        response = parts[1].strip().strip('"').strip("'")
                        return flag, response
        
        # If parsing fails, default to flag 0
        logger.warning(f"Failed to parse Gemini response: {response_text}")
        return 0, response_text
        
    except Exception as e:
        logger.error(f"Error parsing Gemini response: {e}")
        return 0, response_text

async def update_customisations(user_id: str, agraja_doc_id: str, current_instructions: str, instruction: str) -> None:
    """Update customisations in Firestore (runs in background)"""
    try:
        prompt = f"""You are an instruction editor for a chatbot's persistent customization settings. Your task is to update a block of stored instructions based on a new instruction.

Input 1: Existing Customisation Instructions (if any):
{current_instructions}

Input 2: New Instruction (cleaned and ready-to-store):
"{instruction}"

Output Task:
Update the customization instructions using the new instruction, following these rules:

If the new instruction is similar in meaning or scope to an existing one (e.g., both are about the chatbot's reply language, tone, or topic restrictions), completely replace the old one with the new one.

If no similar instruction exists, append the new one to the list.

Always return all customization instructions after the update, including unchanged ones.

The final output should be a clean, well-formatted multi-line list suitable for direct inclusion in a chatbot prompt.

Only update what is relevant — unrelated instructions must remain unchanged.

Output Format:
Return only the full, updated list of customization instructions as a plain multi-line string.
Do not return JSON, explanations, or any extra commentary.

Begin now."""
        
        updated_instructions = await call_gemini_api(prompt, GEMINI_KEY_CUSTOM)
        # updated_instructions = "abcdefghing"
        
        # Update Firestore with correct path: users/{userId}/Agraja/{agrajaDocId}
        doc_ref = db.collection('users').document(user_id).collection('Agraja').document(agraja_doc_id)
        doc_ref.update({'customisations1': updated_instructions})
        
        logger.info(f"Successfully updated customisations for user {user_id}, agraja doc {agraja_doc_id}")
        
    except Exception as e:
        logger.error(f"Failed to update customisations: {e}")

async def update_interests(user_id: str, agraja_doc_id: str, interest_block: str, instruction: str) -> None:
    """Update interests in Firestore (runs in background)"""
    try:
        prompt = f"""You are updating a user interest profile for a chatbot designed to engage elderly users in warm, meaningful, and emotionally intelligent conversations. The chatbot aims to talk like a companion and make the user feel heard, respected, and personally understood. All information you generate will be used to improve how the chatbot responds and what it talks about.

---

**Existing Interests Block (if any):**
{interest_block}

---

**New Instruction (cleaned and ready-to-store):**
"{instruction}"

---

**Goal:**
Update the interest list based on the new instruction. Follow these steps:
1. If the new instruction is closely related to something already present, **merge or enhance** the existing interest.
2. If the new instruction is unrelated, **add it as a new bullet point**.
3. Do not remove or modify unrelated interests.
4. Keep it clean, grouped by theme, and prompt-ready.

---

**Return only the updated Interest Block as plain text.**
Begin now.

---"""
        
        updated_interests = await call_gemini_api(prompt, GEMINI_KEY_CUSTOM)
        
        # Update Firestore with correct path: users/{userId}/Agraja/{agrajaDocId}
        doc_ref = db.collection('users').document(user_id).collection('Agraja').document(agraja_doc_id)
        doc_ref.update({'intrests2': updated_interests})
        
        logger.info(f"Successfully updated interests for user {user_id}, agraja doc {agraja_doc_id}")
        
    except Exception as e:
        logger.error(f"Failed to update interests: {e}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        # Immediately call main Gemini API and return response to user
        chatbot_response = await call_gemini_api(request.prompt, GEMINI_KEY_MAIN)
        
        # Start background classification and update task (non-blocking)
        asyncio.create_task(classify_and_update(
            request.userDoc,
            request.agrajaDocId,
            request.prompt
        ))
        
        # Return the chatbot response immediately without any parsing or processing
        return {
            "response": chatbot_response
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "Agrajabot FastAPI server is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test Firebase connection
        test_doc = db.collection('test').document('health_check')
        test_doc.set({'timestamp': firestore.SERVER_TIMESTAMP}, merge=True)
        
        return {
            "status": "healthy",
            "firebase": "connected",
            "gemini_main_key": "configured" if GEMINI_KEY_MAIN else "missing",
            "gemini_custom_key": "configured" if GEMINI_KEY_CUSTOM else "missing"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@app.get('/test')
async def test():
    try:
        asyncio.create_task(update_customisations("98d5t4DPFnOfUIjuYgz0RCpndFG3", "qHVFbrobYgAor4EGZUNe", "abcd", "abdc"))
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
