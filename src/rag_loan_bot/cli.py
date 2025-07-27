import typer
from rich import print
from .pipeline import RAGPipeline

app = typer.Typer()

@app.command()
def ask(q: str):
    pipe = RAGPipeline()
    out = pipe.query(q)
    print("[bold green]Answer:[/bold green]", out["answer"])
    print("\n[bold]Contexts used:[/bold]")
    for i, c in enumerate(out["contexts"], 1):
        print(f"--- Context {i} ---\n{c[:500]}...\n")

if __name__ == "__main__":
    app()
