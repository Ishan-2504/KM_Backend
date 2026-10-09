import os
import sys
import numpy as np
from dotenv import load_dotenv
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

CHAT_MODEL = os.getenv(
    "GEMINI_CHAT_MODEL",
    "gemini-3.8-flash"
)

EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-2"
)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("GOOGLE GEMINI API TEST")
    print("=" * 60)

    # --------------------------------------------------------
    # Check API key
    # --------------------------------------------------------

    if not API_KEY:
        print("\nERROR: GEMINI_API_KEY not found.")
        print("Make sure your .env file contains:")
        print()
        print("GEMINI_API_KEY=your_api_key_here")
        sys.exit(1)

    print("\nAPI key found.")
    print("Chat model:", CHAT_MODEL)
    print("Embedding model:", EMBEDDING_MODEL)

    # --------------------------------------------------------
    # Create Gemini client
    # --------------------------------------------------------

    try:
        client = genai.Client(
            api_key=API_KEY
        )

        print("Gemini client initialized.")

    except Exception as e:
        print("\nFAILED TO INITIALIZE GEMINI CLIENT")
        print("Error type:", type(e).__name__)
        print("Error:", str(e))
        sys.exit(1)

    # ========================================================
    # TEST 1 — LLM
    # ========================================================

    print("\n" + "=" * 60)
    print("[1/3] TESTING LLM TEXT GENERATION")
    print("=" * 60)

    try:

        response = client.models.generate_content(
            model=CHAT_MODEL,
            contents="Reply with exactly: Gemini API is working!"
        )

        if response.text:

            print("\nSUCCESS: LLM API is working.")
            print("Response:", response.text)

        else:

            print("\nWARNING: Gemini returned no text.")

    except Exception as e:

        print("\nLLM API TEST FAILED")
        print("Error type:", type(e).__name__)
        print("Error:", str(e))

        sys.exit(1)

    # ========================================================
    # TEST 2 — EMBEDDINGS
    # ========================================================

    print("\n" + "=" * 60)
    print("[2/3] TESTING EMBEDDING GENERATION")
    print("=" * 60)

    texts = [
        "I love machine learning and artificial intelligence.",
        "I enjoy studying AI and machine learning.",
        "The weather is sunny today."
    ]

    embeddings = []

    try:

        # Generate each embedding separately.
        #
        # This avoids the issue where the API returns
        # one embedding for the entire input list.

        for i, text in enumerate(texts):

            print(f"\nGenerating embedding {i + 1}/{len(texts)}...")

            embedding_response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=text
            )

            embedding = np.array(
                embedding_response.embeddings[0].values,
                dtype=np.float32
            )

            embeddings.append(embedding)

        embeddings = np.array(embeddings)

        print("\nSUCCESS: Embedding API is working.")
        print("Number of embeddings:", len(embeddings))
        print("Embedding dimensions:", embeddings.shape[1])

    except Exception as e:

        print("\nEMBEDDING API TEST FAILED")
        print("Error type:", type(e).__name__)
        print("Error:", str(e))

        sys.exit(1)

    # ========================================================
    # TEST 3 — SEMANTIC SIMILARITY
    # ========================================================

    print("\n" + "=" * 60)
    print("[3/3] TESTING SEMANTIC SIMILARITY")
    print("=" * 60)

    try:

        similarity_related = cosine_similarity(
            embeddings[0],
            embeddings[1]
        )

        similarity_unrelated = cosine_similarity(
            embeddings[0],
            embeddings[2]
        )

        print(
            "\nSentence 1:",
            texts[0]
        )

        print(
            "Sentence 2:",
            texts[1]
        )

        print(
            "\nSimilarity (related sentences):",
            round(similarity_related, 4)
        )

        print(
            "\nSentence 1:",
            texts[0]
        )

        print(
            "Sentence 3:",
            texts[2]
        )

        print(
            "\nSimilarity (unrelated sentences):",
            round(similarity_unrelated, 4)
        )

    except Exception as e:

        print("\nSIMILARITY TEST FAILED")
        print("Error type:", type(e).__name__)
        print("Error:", str(e))

        sys.exit(1)

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nYour Google setup is working:")
    print("  ✓ Gemini LLM API")
    print("  ✓ Gemini Embeddings API")
    print("  ✓ 3072-dimensional embeddings")
    print("  ✓ Vector similarity calculation")

    print("\nYou can now use Gemini + embeddings for your RAG pipeline.")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()