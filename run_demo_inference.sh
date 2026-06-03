#!/usr/bin/env bash
# Demo inference launcher — search-only pipeline for a custom question.
#
# Usage:
#   bash run_demo_inference.sh
#   bash run_demo_inference.sh "Your custom question here"
#   bash run_demo_inference.sh "Your question" gemini-3-pro-preview

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-xx}"
export OPENAI_API_BASE="${OPENAI_API_BASE:-xx}"
export SERPER_API_KEY="${SERPER_API_KEY:-xx}"
export GITHUB_TOKEN="${GITHUB_TOKEN:-xx}"

QUESTION="${1:-Design a multi-agent pipeline to analyze legal contracts and extract key obligations.}"
MODEL="${2:-gemini-3-pro-preview}"

echo "=========================================="
echo "Unified-MAS Demo Inference (search only)"
echo "=========================================="
echo "Question : ${QUESTION}"
echo "Model    : ${MODEL}"
echo "=========================================="

python demo_inference.py \
  --question "${QUESTION}" \
  --model "${MODEL}"
