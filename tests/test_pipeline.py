from src.rag_loan_bot.pipeline import RAGPipeline

def test_pipeline_runs():
    pipe = RAGPipeline()
    out = pipe.query("What is Loan_Status?")
    assert "answer" in out
    assert len(out["contexts"]) > 0
