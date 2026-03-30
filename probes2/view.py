"""Interactive Gradio viewer for eval parquet files."""

import html
import json
from pathlib import Path
from typing import Any, Optional

import typer

app = typer.Typer(help="Interactive eval parquet viewer.", no_args_is_help=True)

ROLE_COLORS = {
    "system": ("#6b7280", "#f3f4f6"),
    "user": ("#2563eb", "#eff6ff"),
    "assistant": ("#059669", "#f0fdf4"),
}
REASONING_COLOR = ("#8b5cf6", "#faf5ff")


# ---------------------------------------------------------------------------
# Parquet field helpers
# ---------------------------------------------------------------------------


def _coerce_messages(raw: Any) -> list[dict[str, str]]:
    """Extract the message list from the messages column value."""
    if isinstance(raw, dict):
        inner = raw.get("messages", [])
    else:
        inner = raw

    if isinstance(inner, str):
        inner = json.loads(inner)

    return [dict(m) for m in inner]


def _coerce_outputs(raw: Any) -> Optional[dict[str, str]]:
    """Extract the outputs dict from the messages column value."""
    if isinstance(raw, dict):
        outputs = raw.get("outputs")
        if isinstance(outputs, dict):
            return dict(outputs)
    return None


def _format_messages_html(messages_field: Any) -> str:
    msgs = _coerce_messages(messages_field)
    outputs = _coerce_outputs(messages_field)

    parts: list[str] = []
    for msg in msgs:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        border, bg = ROLE_COLORS.get(role, ("#374151", "#f9fafb"))
        parts.append(
            f'<div style="border-left:4px solid {border};padding:8px 12px;margin:8px 0;'
            f'background:{bg};border-radius:4px;color:#111;">'
            f'<strong style="color:{border};">{html.escape(role.upper())}</strong>'
            f'<pre style="white-space:pre-wrap;margin:4px 0 0 0;color:#111;">{html.escape(str(content))}</pre>'
            f"</div>"
        )

    if outputs:
        reasoning = outputs.get("reasoning_content", "")
        text = outputs.get("text", "")
        if reasoning and str(reasoning).strip():
            border, bg = REASONING_COLOR
            parts.append(
                f'<div style="border-left:4px solid {border};padding:8px 12px;margin:8px 0;'
                f'background:{bg};border-radius:4px;color:#111;">'
                f'<strong style="color:{border};">REASONING</strong>'
                f'<pre style="white-space:pre-wrap;margin:4px 0 0 0;color:#111;">{html.escape(str(reasoning))}</pre>'
                f"</div>"
            )
        if text:
            border, bg = ROLE_COLORS["assistant"]
            parts.append(
                f'<div style="border-left:4px solid {border};padding:8px 12px;margin:8px 0;'
                f'background:{bg};border-radius:4px;color:#111;">'
                f'<strong style="color:{border};">OUTPUT</strong>'
                f'<pre style="white-space:pre-wrap;margin:4px 0 0 0;color:#111;">{html.escape(str(text))}</pre>'
                f"</div>"
            )

    return "\n".join(parts) if parts else "<p>No messages found.</p>"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    path: str = typer.Argument(
        ..., help="Path to a parquet file or HF Hub dataset id (e.g. tytodd/my-eval)"
    ),
    subset: Optional[str] = typer.Option(
        None,
        "--subset",
        "-s",
        help="Dataset subset/config name — when provided, path is treated as an HF Hub dataset id",
    ),
    port: int = typer.Option(7860, "--port", "-p", help="Server port"),
    share: bool = typer.Option(False, "--share", help="Create a public Gradio link"),
) -> None:
    """Launch an interactive viewer for an eval parquet file or HF Hub dataset."""
    try:
        import gradio as gr
    except ImportError:
        print('gradio is not installed. Run: uv pip install -e ".[view]"')
        raise typer.Exit(1)

    import pandas as pd

    if subset is not None:
        from datasets import load_dataset as hf_load_dataset

        print(f"Loading HF dataset {path} (subset={subset}) ...")
        ds = hf_load_dataset(path, subset)
        split_name = list(ds.keys())[0]
        df = ds[split_name].to_pandas()
        display_name = f"{path}/{subset}"
    else:
        local_path = Path(path)
        if not local_path.exists():
            print(f"File not found: {local_path}")
            raise typer.Exit(1)
        # Read schema first to detect heavy list-of-float columns (e.g. embeddings)
        # and skip them at read time so we never load them into memory.
        import pyarrow.parquet as pq

        schema = pq.read_schema(local_path)
        heavy_cols: dict[str, str] = {}  # col -> placeholder
        light_cols: list[str] = []
        for field in schema:
            import pyarrow as pa

            if pa.types.is_list(field.type) or pa.types.is_large_list(field.type):
                # Peek at first non-null value to get dimension
                col_table = pq.read_table(local_path, columns=[field.name]).column(0)
                for val in col_table:
                    v = val.as_py()
                    if v is not None and len(v) > 20:
                        heavy_cols[field.name] = f"[{len(v)}-dim vector]"
                        break
                else:
                    light_cols.append(field.name)
            else:
                light_cols.append(field.name)

        df = pd.read_parquet(local_path, columns=light_cols)
        for col, placeholder in heavy_cols.items():
            df[col] = placeholder
        display_name = local_path.name

    # Build display dataframe — show all columns, stringify complex types,
    # skip "messages" (viewed via button).
    display_df = pd.DataFrame()
    for col in df.columns:
        if col == "messages":
            continue
        sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
        if sample is not None and isinstance(sample, (dict, list)):
            display_df[col] = df[col].apply(
                lambda x: json.dumps(x, default=str) if x is not None else ""
            )
        else:
            display_df[col] = df[col]

    # --- Gradio UI ---

    with gr.Blocks(title=f"Eval Viewer \u2014 {display_name}") as demo:
        gr.Markdown(f"# Eval Viewer \u2014 `{display_name}`\n{len(df)} rows")

        selected_row = gr.State(-1)

        with gr.Row(equal_height=False):
            with gr.Column(scale=2):
                table = gr.Dataframe(
                    value=display_df,
                    label="Dataset",
                    interactive=False,
                )

            with gr.Column(scale=1):
                show_btn = gr.Button("Show Messages", visible=False)
                messages_html = gr.HTML(
                    value="<p style='color:#6b7280;'>Click a row to view details.</p>",
                )

        def on_row_select(evt: gr.SelectData):
            row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else evt.index
            return (
                row_idx,
                gr.Button(visible=has_messages),
                gr.HTML(value="<p style='color:#6b7280;'>Click <b>Show Messages</b> to view.</p>"),
            )

        def on_show_messages(row_idx):
            if row_idx is None or row_idx < 0 or not has_messages:
                return gr.HTML(value="<p>No messages.</p>")
            row = df.iloc[row_idx]
            return gr.HTML(value=_format_messages_html(row["messages"]))

        table.select(
            fn=on_row_select,
            inputs=[],
            outputs=[selected_row, show_btn, messages_html],
        )

        show_btn.click(
            fn=on_show_messages,
            inputs=[selected_row],
            outputs=[messages_html],
        )

    demo.launch(server_port=port, share=share)


if __name__ == "__main__":
    app()
