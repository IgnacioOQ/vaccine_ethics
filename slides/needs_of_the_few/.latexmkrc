# Self-contained latexmk config — inlined from latex_repo's root
# .latexmkrc by latex_mcp/latex_get_template so this template can
# be dropped into an external paper repo and compile standalone
# (no parent .latexmkrc required).

# Engine: 1=pdflatex, 4=lualatex, 5=xelatex
# Locked to lualatex per NOTES.md.
$pdf_mode = 4;

$lualatex = 'lualatex -synctex=1 -interaction=nonstopmode -file-line-error %O %S';

# Use biber (not legacy bibtex) for biblatex bibliographies.
$bibtex_use = 2;

$aux_dir = 'build';
$out_dir = '.';
