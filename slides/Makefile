# Build the LaTeX decks. Two passes: the second resolves \inserttotalframenumber.
#
# Requires only texlive-latex-recommended + pgf (beamer, tikz, pgfplots,
# listings, booktabs). No contributed themes, no tcolorbox, no fontawesome --
# the decks must build on a bare TeX Live and in CI.

LATEX   ?= pdflatex
LFLAGS   = -interaction=nonstopmode -halt-on-error -file-line-error
PY      ?= python3
TEX      = $(wildcard *.tex)
PDF      = $(TEX:.tex=.pdf)

.PHONY: all figures clean distclean
all: $(PDF)

figures:
	$(PY) figures/make_figures.py --data ../research/data --out figures

%.pdf: %.tex common/pdmtheme.sty | figures
	$(LATEX) $(LFLAGS) $< >/dev/null || { $(LATEX) -interaction=nonstopmode $< | tail -40; false; }
	$(LATEX) $(LFLAGS) $< >/dev/null
	@echo "built $@ ($$($(PY) -c "import sys;print(sys.argv[1])" $$(stat -c%s $@)) bytes)"

clean:
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.vrb *.synctex.gz

distclean: clean
	rm -f $(PDF) figures/*.pdf figures/*.png
