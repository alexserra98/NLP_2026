from IPython.display import display, HTML
import html as html_mod
import pandas as pd

def show_tokens(text, tokenizer):
    """Display tokenization as colored spans, works in both light and dark mode."""

    tokens = tokenizer.tokenize(text)
    # Pairs of (background, text-color) that are readable on any IDE theme
    palette = [
        ("#a8d8ea", "#1a1a2e"),  # light blue
        ("#f8b4c8", "#1a1a2e"),  # pink
        ("#b5e6b5", "#1a1a2e"),  # green
        ("#ffe5a0", "#1a1a2e"),  # yellow
        ("#d4bbf0", "#1a1a2e"),  # lavender
        ("#ffd6b0", "#1a1a2e"),  # peach
    ]
    spans = ""
    for i, tok in enumerate(tokens):
        bg, fg = palette[i % len(palette)]
        # Replace BPE markers: Ġ/▁ = space shown as _, Ċ = newline shown as \n
        display_tok = tok.replace("Ġ", "_").replace("▁", "_")
        display_tok = display_tok.replace("Ċ", "\\n").replace("\n", "\\n")
        # HTML-escape so angle brackets etc. render safely
        display_tok = html_mod.escape(display_tok)
        spans += (
            f'<span style="background:{bg}; color:{fg}; padding:2px 5px; '
            f'border-radius:4px; margin:1px; display:inline-block; '
            f'font-family:monospace; font-size:13px; white-space:pre;">{display_tok}</span>'
        )
    display(HTML(
        f'<div style="line-height:2.4;">'
        f'<b style="color:inherit;">{len(tokens)} tokens:</b><br>{spans}</div>'
    ))


#  Helper function for nice visualization.
def color_label_table(df, column, colors={"positive": "green", "negative": "red"}):
    """Display a DataFrame with colored text labels in the specified column."""
    return df.style.applymap(
        lambda v: f"color: {colors[v]}; font-weight: bold" if v in colors else "",
        subset=[column],
    )    