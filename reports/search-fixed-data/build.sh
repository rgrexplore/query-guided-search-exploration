#!/usr/bin/env bash
# Build the fixed-dataset report. Figures and tables come from make_results.py.
set -euo pipefail
REPORT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$REPORT_DIR/../.." && pwd)"
mkdir -p "$REPORT_DIR/build" "$PROJECT_DIR/output/pdf"
cd "$REPORT_DIR"
for pass in 1 2 3; do
  if ! pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory=build report.tex > "build/compile-$pass.txt"; then
    tail -n 40 "build/compile-$pass.txt"
    exit 1
  fi
done
cp build/report.pdf "$PROJECT_DIR/output/pdf/search-methods-fixed-data.pdf"
echo "wrote $PROJECT_DIR/output/pdf/search-methods-fixed-data.pdf"
