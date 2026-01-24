#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT_DIR"

echo "Running tests with coverage (overall: --cov=app)..."

pytest -q \
  --cov=app \
  --cov-report=term-missing \
  --cov-report=xml:coverage.xml \
  --cov-report=html:htmlcov

echo
echo "Coverage artifacts generated:"
echo "  - $ROOT_DIR/coverage.xml"
echo "  - $ROOT_DIR/htmlcov/index.html"
