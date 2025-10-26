cd .venv\Scripts
call activate
cd ..
cd ..
cd src\eval
python kg_rag_eval.py "..\..\data\generated_qna_large_gpt-oss20b_junior_on_2025-10-17.csv"
python kg_rag_eval.py "..\..\data\generated_qna_large_gpt-oss20b_medior_on_2025-10-17.csv"
python kg_rag_eval.py "..\..\data\generated_qna_large_gpt-oss20b_senior_on_2025-10-17.csv"
python kg_rag_eval.py "..\..\data\stackowerQnA_context.csv"