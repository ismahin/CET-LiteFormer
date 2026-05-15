# CET-LiteFormer architecture figures

## Files

| File | Description |
|------|-------------|
| **`cet_liteformer_architecture_rich.mmd`** | **Recommended** — color-coded, HTML labels, legend, paper-style |
| `cet_liteformer_architecture.mmd` | Plain technical version |
| `cet_liteformer_architecture.html` | Browser preview (needs local server or use Mermaid Live) |
| `cet_liteformer_architecture_v2.png` | Raster preview |

## Render rich diagram (best quality)

1. Open **[mermaid.live](https://mermaid.live)**
2. Paste contents of **`cet_liteformer_architecture_rich.mmd`**
3. **Actions → Export → SVG**
4. Convert SVG → PDF (Inkscape, Illustrator, or `rsvg-convert -f pdf -o fig.pdf fig.svg`)
5. In LaTeX: `\includegraphics[width=\textwidth]{figures/cet_liteformer_architecture.pdf}`

## Rich diagram features

- Title banner + **color legend** (Data / Tokens / Gate / Encoder / Output / Loss)
- **Stage-colored subgraphs** (blue → teal → green → orange → purple)
- **HTML labels** with subscripts and bold equations
- **Thick pipeline arrows** (`==>`) for main data flow
- **Dashed arrows** for MI prior, losses, early exit
- **Stadium / cylinder / diamond** node shapes for inputs and decisions
- Icons (📥 🧹 🔍 etc.) for quick visual parsing

## Local HTML preview

```powershell
cd D:\Research\CET-LiteFormer\figures
python -m http.server 8080
# Browser: http://localhost:8080/cet_liteformer_architecture.html
```

## VS Code

Install **Markdown Preview Mermaid Support** or **Mermaid Chart** extension → open `.mmd` file → preview.
