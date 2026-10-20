import os
# Uncomment the next line ONLY if you still get '@' garbage output.
# Then fully quit Ollama (system tray) and restart it before running this script.
# os.environ["OLLAMA_GPU_LAYERS"] = "0"   # force CPU mode

import requests
import re

# SECTION 1: LLM configuration
MODEL = "llama3.2"
URL = "http://localhost:11434/api/chat"


# SECTION 2: Communicate with the LLM
def ask_llm(prompt):
    response = requests.post(
        URL,
        json={
            "model": MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "keep_alive": 0,            # unload after each request (fixes @ garbage)
            "options": {
                "num_predict": 1024,
                "temperature": 0.4
            }
        },
        timeout=300
    )

    response.raise_for_status()

    content = response.json()["message"]["content"]

    # Detect real garbage output
    if re.search(r"@{5,}", content):
        print("\n[ERROR] Model produced '@' garbage output.")
        print("Fix steps:")
        print("  1. Quit Ollama from the system tray and restart it.")
        print("  2. Uncomment the OLLAMA_GPU_LAYERS line at the top of this script.")
        print("  3. Re-run this script (Ollama must be restarted AFTER setting it).")
        print("  4. If still broken: update NVIDIA driver (>=572.60) and Ollama.")
        print("  5. Last resort: ollama rm llama3.2 && ollama pull llama3.2\n")
        return content  # still return so the file is written for inspection

    return content


# SECTION 3: Create a research plan
def create_plan(question):
    prompt = f"""
    You are an AI research assistant.

    Create a research plan for this question:
    {question}

    Include:
    1. Research objectives
    2. Keywords
    3. Important topics
    4. Possible research gaps

    Do not invent scientific references.
    """
    return ask_llm(prompt)


# SECTION 4: Create a research report
def create_report(question, plan):
    prompt = f"""
    Write a preliminary research report about:
    {question}

    Use this research plan:
    {plan}

    Organize the report into:
    1. Introduction
    2. Main concepts
    3. Possible methods
    4. Potential research gaps
    5. Conclusion

    Do not invent scientific references.
    Explain that claims and research gaps
    must be verified using scientific papers.
    """
    return ask_llm(prompt)


# SECTION 5: Main program
def main():
    # Warm-up: this loads the model once so the first real request isn't slow.
    # With keep_alive=0, the model unloads after this call — that's expected.
    print("Testing model (warm-up)...")
    try:
        test = ask_llm("Say the word OK.")
        print(f"Model replied: {test[:80]!r}\n")
    except Exception as e:
        print(f"Warm-up failed: {e}\n")

    question = input("Enter your research question: ")

    print("\nCreating research plan...\n")
    plan = create_plan(question)
    print(plan)

    print("\nCreating research report...\n")
    report = create_report(question, plan)
    print(report)

    # Save the report to a text file
    with open("research_report.txt", "w", encoding="utf-8") as file:
        file.write("RESEARCH QUESTION\n")
        file.write(question + "\n\n")

        file.write("RESEARCH PLAN\n")
        file.write(plan + "\n\n")

        file.write("RESEARCH REPORT\n")
        file.write(report)

    print("\nReport saved as research_report.txt")


# SECTION 6: Start the program
if __name__ == "__main__":
    main()