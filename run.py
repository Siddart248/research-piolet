"""Project execution entrypoint for ResearchPilot.

Provides CLI commands to:
1. Run the FastAPI server: `python run.py --serve`
2. Run the LangGraph ResearchPilot agent: `python run.py --agent "..."`
3. Run the evaluation benchmark: `python run.py --eval`
4. Test RAG retrieval directly: `python run.py --query "..."`
5. Rebuild local knowledge base vector index: `python run.py --build-index`
"""

import argparse
import sys
import uvicorn

from app.config import get_settings
from app.evaluation import run_evaluation
from app.graph import run_research_agent
from app.logging_config import setup_logging
from app.retrieval import LocalVectorStore, build_index


def main() -> None:
    setup_logging("INFO")

    parser = argparse.ArgumentParser(description="ResearchPilot CLI Runner")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI web service with Uvicorn")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address for API server")
    parser.add_argument("--port", type=int, default=8000, help="Port for API server")
    parser.add_argument("--agent", type=str, default=None, help="Execute full LangGraph ResearchPilot agent on query")
    parser.add_argument("--query", type=str, default=None, help="Query the local retrieval index directly")
    parser.add_argument("--build-index", action="store_true", help="Build or rebuild the local vector index")
    parser.add_argument("--eval", action="store_true", help="Run engineering evaluation test suite")
    parser.add_argument("--threshold", type=float, default=None, help="Custom similarity threshold")
    parser.add_argument("--top-k", type=int, default=3, help="Top-K results to retrieve")

    args = parser.parse_args()

    # Mode 1: Serve FastAPI
    if args.serve:
        print(f"[*] Starting ResearchPilot FastAPI service on http://{args.host}:{args.port}")
        uvicorn.run("app.main:app", host=args.host, port=args.port, reload=False)
        return

    # Mode 2: Run Evaluation
    if args.eval:
        print("[*] Running ResearchPilot engineering evaluation framework...")
        res = run_evaluation()
        print("\n" + "=" * 60)
        print("RESEARCHPILOT EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Overall Pass Rate            : {res['metrics']['overall_pass_rate'] * 100:.1f}% ({res['passed_cases']}/{res['total_test_cases']})")
        print(f"Structured Output Validity   : {res['metrics']['structured_output_validity'] * 100:.1f}%")
        print(f"Tool Usage Accuracy          : {res['metrics']['tool_usage_accuracy'] * 100:.1f}%")
        print(f"Retrieval Success Rate       : {res['metrics']['retrieval_success_rate'] * 100:.1f}%")
        print(f"Insufficient Context Accuracy: {res['metrics']['insufficient_context_accuracy'] * 100:.1f}%")
        print(f"Failure Taxonomy Breakdown   : {res['failure_taxonomy_breakdown']}")
        print("=" * 60)
        return

    # Mode 3: Rebuild Index
    if args.build_index:
        print("[*] Building local vector store from knowledge base...")
        build_index()
        print("[+] Index build complete.")
        return

    # Mode 4: Full LangGraph Agent Execution
    if args.agent:
        print("=" * 60)
        print("RESEARCHPILOT AGENT INVOCATION")
        print("=" * 60)
        print(f"Query: {args.agent}\n")

        response = run_research_agent(args.agent)

        print("\n" + "=" * 60)
        print("STRUCTURED RESEARCH RESPONSE (Pydantic)")
        print("=" * 60)
        print(f"Answer:\n{response.answer}\n")
        print("Key Points:")
        for pt in response.key_points:
            print(f"  • {pt}")
        print(f"\nCited Sources        : {response.sources}")
        print(f"Confidence Score     : {response.confidence}")
        print(f"Insufficient Context : {response.insufficient_context}")
        print(f"Tools Invoked        : {response.tool_calls}")
        print("=" * 60)
        return

    # Mode 5: Direct Retrieval Query
    if args.query:
        try:
            store = LocalVectorStore.load()
        except FileNotFoundError:
            print("[!] Index not found. Building index first...")
            store = build_index()

        print(f"[*] Executing direct retrieval query: {args.query!r}")
        result = store.retrieve(
            query=args.query,
            top_k=args.top_k,
            threshold=args.threshold,
        )

        print("\n" + "=" * 50)
        print("DIRECT RETRIEVAL RESULT")
        print("=" * 50)
        print(f"Query                : {result.query}")
        print(f"Best Similarity Score: {result.best_similarity}")
        print(f"Threshold Applied    : {result.threshold}")
        print(f"Insufficient Context : {result.insufficient_context}")
        print(f"Qualifying Chunks    : {len(result.chunks)}")

        if result.insufficient_context:
            print("\n[!] Context is insufficient. Refused to return irrelevant chunks.")
        else:
            for i, chunk in enumerate(result.chunks, 1):
                print(f"\n--- Chunk {i} [{chunk.source} | Score: {chunk.similarity}] ---")
                print(chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""))
        return

    print("ResearchPilot CLI. Available flags:")
    print("  --serve               : Launch FastAPI web API on 127.0.0.1:8000")
    print("  --agent '<query>'     : Run full LangGraph AI research agent")
    print("  --eval                : Run engineering evaluation framework")
    print("  --query '<query>'     : Direct RAG retrieval query")
    print("  --build-index         : Rebuild local knowledge base vector index")


if __name__ == "__main__":
    main()
