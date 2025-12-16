# Testsets for Ragas (Amber)

Place evaluation datasets here. Recommended files:
- `amber_carbonio_ragas_eval_template.jsonl` / `.csv`
- `amber_carbonio_ragas_gold_testset.jsonl` / `.csv`

Format per row (JSONL or CSV):
- `user_input`: question
- `reference`: ground truth answer
- `retrieved_contexts`: optional list (leave empty if runner should call backend)
- `response`: optional prefilled answer (runner can fill after calling backend)
- `metadata`: JSON object or string with `intent` (admin|user), `source_doc`, `source_pages`, `qa_type`
