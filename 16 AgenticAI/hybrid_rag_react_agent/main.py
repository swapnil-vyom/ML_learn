from app.agent.agentic import AgenticSession

def main():
    session = AgenticSession()
    while True:
        q = input("\nAsk a question (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        session.run_query(q)

if __name__ == "__main__":
    main()
