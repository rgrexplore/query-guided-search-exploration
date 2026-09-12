#!/usr/bin/env bash
set -euo pipefail
REPORT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$REPORT_DIR/../.." && pwd)"
BUILD_DIR="$REPORT_DIR/build"
mkdir -p "$BUILD_DIR" "$PROJECT_DIR/output/pdf"
cd "$REPORT_DIR"
for pass in 1 2 3; do
  if ! pdflatex -interaction=nonstopmode -halt-on-error -file-line-error -output-directory="$BUILD_DIR" report.tex > "$BUILD_DIR/compile-$pass.txt"; then
    tail -n 55 "$BUILD_DIR/compile-$pass.txt"
    exit 1
  fi
done
cp "$BUILD_DIR/report.pdf" "$PROJECT_DIR/output/pdf/search-methods-mathematics.pdf"
printf 'Built %s\n' "$PROJECT_DIR/output/pdf/search-methods-mathematics.pdf"
