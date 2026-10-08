"""Engineering evaluation framework and failure analysis for ResearchPilot.

Provides:
- Automated evaluation runner against evaluation/test_cases.json
- Metric calculation:
  - retrieval_success
  - tool_usage_correct
  - structured_output_valid
  - insufficient_context_handled
  - overall_pass_rate
- Failure root-cause categorization:
  - retrieval
  - tool_selection
  - prompt/context
  - llm_output
  - structured_validation
  - application_logic
"""

import json
import logging
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT
from app.graph import run_research_agent
from app.schemas import ResearchResponse

logger = logging.getLogger("evaluation")

DEFAULT_TEST_CASES_PATH = PROJECT_ROOT / "evaluation" / "test_cases.json"
DEFAULT_SUMMARY_PATH = PROJECT_ROOT / "evaluation" / "latest_summary.json"


def evaluate_single_case(test_case: dict[str, Any]) -> dict[str, Any]:
    """Execute a single evaluation test case and assess compliance across criteria."""
    case_id = test_case["id"]
    case_name = test_case["name"]
    query = test_case["query"]
    context_turns = test_case.get("context_turns", [])
    expected_tools = set(test_case.get("expected_tools", []))
    expected_sources = set(test_case.get("expected_sources", []))
    expect_insufficient = test_case.get("expect_insufficient_context", False)
    expected_keywords = test_case.get("expected_content_keywords", [])

    logger.info(f"[EVAL] Running {case_id}: {case_name!r}")

    failure_reason: str | None = None
    failure_category: str | None = None
    structured_valid = False
    tools_correct = False
    retrieval_success = True
    insufficient_correct = False
    content_matches = True

    try:
        response: ResearchResponse = run_research_agent(query=query, history=context_turns)

        # 1. Structured output validity
        if isinstance(response, ResearchResponse) and 0.0 <= response.confidence <= 1.0:
            structured_valid = True
        else:
            failure_category = "structured_validation"
            failure_reason = "Model output did not conform to ResearchResponse contract."

        # 2. Tool usage correctness
        actual_tools = set(response.tool_calls)
        if actual_tools == expected_tools:
            tools_correct = True
        else:
            if not failure_category:
                failure_category = "tool_selection"
                failure_reason = f"Tool mismatch: expected {expected_tools}, got {actual_tools}"

        # 3. Insufficient context handling
        if response.insufficient_context == expect_insufficient:
            insufficient_correct = True
        else:
            if not failure_category:
                failure_category = "prompt/context" if not expect_insufficient else "retrieval"
                failure_reason = (
                    f"Insufficient context mismatch: expected {expect_insufficient}, got {response.insufficient_context}"
                )

        # 4. Retrieval source accuracy
        if expected_sources:
            actual_sources = set(response.sources)
            if expected_sources.issubset(actual_sources):
                retrieval_success = True
            else:
                retrieval_success = False
                if not failure_category:
                    failure_category = "retrieval"
                    failure_reason = f"Source mismatch: expected {expected_sources}, got {actual_sources}"

        # 5. Content check
        answer_lower = response.answer.lower()
        missing_kw = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
        if missing_kw and not expect_insufficient:
            content_matches = False
            if not failure_category:
                failure_category = "llm_output"
                failure_reason = f"Missing expected key phrases: {missing_kw}"

        case_passed = bool(
            structured_valid
            and tools_correct
            and insufficient_correct
            and retrieval_success
            and content_matches
        )

        return {
            "id": case_id,
            "name": case_name,
            "category": test_case.get("category"),
            "passed": case_passed,
            "structured_output_valid": structured_valid,
            "tool_usage_correct": tools_correct,
            "retrieval_success": retrieval_success,
            "insufficient_context_handled": insufficient_correct,
            "failure_category": None if case_passed else failure_category,
            "failure_reason": None if case_passed else failure_reason,
            "actual_response": {
                "answer_snippet": response.answer[:160] + "...",
                "sources": response.sources,
                "tool_calls": response.tool_calls,
                "confidence": response.confidence,
                "insufficient_context": response.insufficient_context,
            },
        }

    except Exception as err:
        logger.error(f"[EVAL] Exception in {case_id}: {err}")
        return {
            "id": case_id,
            "name": case_name,
            "category": test_case.get("category"),
            "passed": False,
            "structured_output_valid": False,
            "tool_usage_correct": False,
            "retrieval_success": False,
            "insufficient_context_handled": False,
            "failure_category": "application_logic",
            "failure_reason": f"Unhandled runtime error: {type(err).__name__}: {err}",
            "actual_response": None,
        }


def run_evaluation(
    test_cases_path: Path | str | None = None,
    summary_path: Path | str | None = None,
) -> dict[str, Any]:
    """Execute complete evaluation suite and compile metrics with failure analysis."""
    input_path = Path(test_cases_path) if test_cases_path else DEFAULT_TEST_CASES_PATH
    output_path = Path(summary_path) if summary_path else DEFAULT_SUMMARY_PATH

    if not input_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found at {input_path}")

    raw_cases = json.loads(input_path.read_text(encoding="utf-8"))
    total_cases = len(raw_cases)

    logger.info(f"[EVAL] Starting evaluation across {total_cases} test cases...")

    results: list[dict[str, Any]] = []
    failure_breakdown: dict[str, int] = {
        "retrieval": 0,
        "tool_selection": 0,
        "prompt/context": 0,
        "llm_output": 0,
        "structured_validation": 0,
        "application_logic": 0,
    }

    for tc in raw_cases:
        res = evaluate_single_case(tc)
        results.append(res)
        if not res["passed"] and res["failure_category"]:
            cat = res["failure_category"]
            failure_breakdown[cat] = failure_breakdown.get(cat, 0) + 1

    passed_count = sum(1 for r in results if r["passed"])
    structured_valid_count = sum(1 for r in results if r["structured_output_valid"])
    tools_correct_count = sum(1 for r in results if r["tool_usage_correct"])
    retrieval_success_count = sum(1 for r in results if r["retrieval_success"])
    insufficient_handled_count = sum(1 for r in results if r["insufficient_context_handled"])

    summary = {
        "total_test_cases": total_cases,
        "passed_cases": passed_count,
        "failed_cases": total_cases - passed_count,
        "metrics": {
            "overall_pass_rate": round(passed_count / total_cases, 4) if total_cases > 0 else 0.0,
            "structured_output_validity": round(structured_valid_count / total_cases, 4) if total_cases > 0 else 0.0,
            "tool_usage_accuracy": round(tools_correct_count / total_cases, 4) if total_cases > 0 else 0.0,
            "retrieval_success_rate": round(retrieval_success_count / total_cases, 4) if total_cases > 0 else 0.0,
            "insufficient_context_accuracy": round(insufficient_handled_count / total_cases, 4) if total_cases > 0 else 0.0,
        },
        "failure_taxonomy_breakdown": failure_breakdown,
        "detailed_results": results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"[EVAL] Completed evaluation. Pass rate: {summary['metrics']['overall_pass_rate'] * 100:.1f}%. Summary saved to {output_path}")

    return summary


if __name__ == "__main__":
    from app.logging_config import setup_logging
    setup_logging("INFO")
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
