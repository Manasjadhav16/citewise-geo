# """
# Standalone Gemini Embedding Test Script
# ---------------------------------------
# This file is fully independent from the main project.
# Run it directly:
#   python gemini_embed_test.py
# """

# import os
# import asyncio
# from google import genai

# MODEL = "models/text-embedding-004"



# async def test_embed(text: str):
#     # api_key = os.getenv("GEMINI_API_KEY")
#     api_key = "your_gemini_api_key_here"

#     if not api_key:
#         print("❌ ERROR: GEMINI_API_KEY missing in environment!")
#         return

#     print(f"\n🔑 Using API Key: {api_key[:6]}********")

#     client = genai.Client(api_key=api_key)

#     try:
#         # Use correct SDK method
#         res = await asyncio.to_thread(
#             client.models.embed_content,
#             model=MODEL,
#             contents=text
#         )

#         print("\n📥 RAW RESPONSE:")
#         print(res)

#         # Extract embedding
#         vec = getattr(res, "embedding", None)
#         vals = getattr(vec, "values", None)

#         if not vals:
#             print("\n❌ ERROR: Empty or missing embedding!")
#             return

#         floats = [float(v) for v in vals]

#         print("\n📊 EMBEDDING DETAILS:")
#         print(f"Length: {len(floats)}")
#         print(f"First 10 values: {floats[:10]}")

#         print("\n✅ SUCCESS — VALID EMBEDDING RECEIVED")

#     except Exception as e:
#         print("\n❌ EXCEPTION:")
#         print(e)


# if __name__ == "__main__":
#     print("Gemini Embedding Quick Test")
#     print("---------------------------")

#     sample_text = "Hello! This is an embedding test using Gemini's text-embedding-004 model."

#     asyncio.run(test_embed(sample_text))


# app/test_embed.py

import os
from google import genai

print("Gemini Embedding Quick Test")
print("---------------------------")

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("❌ No GEMINI_API_KEY found in environment!")
    exit(1)

print(f"\n🔑 Using API Key: {api_key[:10]}********")

client = genai.Client(api_key=api_key)

text = "Hello, this is an embedding test for Gemini."

print("\n📤 Sending text for embedding:", text)

try:
    # Correct SDK call (Jan 2025) using embed_text
    res = client.embed_text(
        model="models/text-embedding-004",
        text=text
    )

    # Extract values correctly: res.embedding.values
    emb = getattr(res, "embedding", None)
    values = getattr(emb, "values", None)
    if not values:
        print("\n❌ ERROR: Empty or missing embedding!")
        exit(1)

    vec = [float(v) for v in values]

    print(f"\n✅ Embedding dimension: {len(vec)}")
    print(f"🔹 First 10 values: {vec[:10]}")

except Exception as e:
    print(f"\n❌ Exception while embedding: {e}")
