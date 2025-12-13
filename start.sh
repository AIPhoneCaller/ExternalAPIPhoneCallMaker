#!/bin/bash
# Quick Start Script for Hebrew Phone Call Assistant

echo "🚀 Starting Hebrew Phone Call Assistant..."
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: python3 -m venv venv"
    exit 1
fi

# Activate venv
source venv/bin/activate

# Check OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    # Try loading from .env
    if [ -f ".env" ]; then
        export $(cat .env | grep -v '^#' | xargs)
    fi
    
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "⚠️  Warning: OPENAI_API_KEY not set!"
        echo "Please set it in .env file"
        echo ""
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
fi

echo "✅ Environment ready"
echo "✅ Starting phone call system..."
echo ""
echo "📞 Speak in Hebrew or English"
echo "🛑 Say 'exit', 'bye', 'סיים', 'צא', or 'סטופ' to end"
echo ""

# Run main.py
python main.py
