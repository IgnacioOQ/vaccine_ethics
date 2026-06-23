# latex_draft/.latexmkrc — standalone build config for this paper.
#
# latexmk does NOT search parent directories for a .latexmkrc (see
# kb_mcp://content/how-to/LATEX_WRITING_SKILL.md). Without this file the
# editor / bare `latexmk` fall back to DVI mode, where PNG figures fail with
# "Cannot determine size of graphic". Setting an explicit PDF engine fixes it.

$pdf_mode  = 1;       # 1 = pdflatex -> PDF (matches this paper's hyperref/pdf setup)
$bibtex_use = 2;      # biber, for the biblatex bibliography (1refs.bib)

# aux-vs-out split: intermediates into build/ (gitignored), final PDF next to
# main.tex (tracked in git). See LATEX_WRITING_SKILL.md.
$aux_dir = 'build';
$out_dir = '.';
