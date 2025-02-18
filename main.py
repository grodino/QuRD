from scripts import run, hf_hub, analyze
from typer import Typer

app = Typer(no_args_is_help=True, add_completion=False)
app.add_typer(run.app, name="bench")
app.add_typer(hf_hub.app, name="hf-hub")
app.add_typer(analyze.app, name="analyze")

if __name__ == "__main__":
    app()
