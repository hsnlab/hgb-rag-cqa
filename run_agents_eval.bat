cd .venv\Scripts
call activate
cd ..\..\src\eval

python agentic_eval.py "..\..\data\generated_qna_large_gpt-oss20b_junior_on_2025-10-17.csv"
python agentic_eval.py "..\..\data\generated_qna_large_gpt-oss20b_medior_on_2025-10-17.csv"
python agentic_eval.py "..\..\data\generated_qna_large_gpt-oss20b_senior_on_2025-10-17.csv"
python agentic_eval.py "..\..\data\stackowerQnA_context.csv"
