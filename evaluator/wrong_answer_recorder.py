import os
from typing import List, Optional

import pandas as pd


class WrongAnswerRecorder:
    """
    Collects wrong answers during evaluation and writes them to an Excel report.
    """

    def __init__(self, output_path: Optional[str] = None):
        default_path = os.getenv("WRONG_ANSWER_REPORT", "logs/wrong_answers.xlsx")
        self.output_path = output_path or default_path
        self.records: List[dict] = []

    def record(
        self,
        *,
        strategy: str,
        question: str,
        expected: str,
        actual: str,
        rationale: str,
        lexical_context: str,
        semantic_context: str,
        judge_prompt: Optional[str],
        expected_file_name: str,
    ):
        self.records.append(
            {
                "strategy": strategy,
                "question": question,
                "expected_answer": expected,
                "actual_answer": actual,
                "rationale": rationale,
                "lexical_results": lexical_context,
                "semantic_results": semantic_context,
                "judge_prompt": judge_prompt or "",
                "expected_file_name": expected_file_name,
            }
        )

    def save(self):
        if not self.records:
            print("No wrong answers recorded; skipping report generation.")
            return

        output_dir = os.path.dirname(self.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        df = pd.DataFrame(self.records)
        df.to_excel(self.output_path, index=False)
        print(
            f"Saved wrong answer report with {len(self.records)} entries to {self.output_path}"
        )

