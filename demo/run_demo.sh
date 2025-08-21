#!/bin/bash

# Mythic AI Engine - LLM to TTS Demo Runner
# This script runs the enhanced demo with various performance profiles
# 
# Project Structure:
# - This script is located in: demo/run_demo.sh
# - Main demo files are in: demo/
# - Output directory: demo/output/
# - Runtime files: demo/runtimes/

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${BLUE}Mythic AI Engine - LLM to TTS Demo${NC}"
echo "=========================================="

# Check if virtual environment exists
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo -e "${YELLOW}Virtual environment not found. Setting up...${NC}"
    cd "$SCRIPT_DIR"
    ./demo_setup.sh
fi

# Activate virtual environment
echo -e "${BLUE}Activating virtual environment...${NC}"
source "$SCRIPT_DIR/.venv/bin/activate"

# Check if required files exist
if [ ! -f "$SCRIPT_DIR/llm_to_tts_demo_enhanced.py" ]; then
    echo -e "${RED}Error: Enhanced demo script not found!${NC}"
    exit 1
fi

if [ ! -f "$SCRIPT_DIR/config.py" ]; then
    echo -e "${RED}Error: Configuration file not found!${NC}"
    exit 1
fi

# Function to show usage
show_usage() {
    echo -e "\n${BLUE}Usage:${NC}"
    echo "  $0 [OPTION]"
    echo ""
    echo -e "${BLUE}Options:${NC}"
    echo "  (no args)       Run enhanced demo with balanced profile (default)"
    echo "  interactive     Run interactive demo mode"
    echo "  fast            Run with fast performance profile"
    echo "  balanced        Run with balanced performance profile"
    echo "  quality         Run with quality performance profile"
    echo "  help            Show this help message"
    echo ""
    echo -e "${BLUE}Examples:${NC}"
    echo "  $0                    # Run enhanced demo (balanced profile)"
    echo "  $0 fast              # Run with fast profile"
    echo "  $0 quality           # Run with quality profile"
    echo "  $0 interactive       # Run interactive mode"
    echo ""
}

# Function to run enhanced demo with default profile
run_default() {
    echo -e "\n${GREEN}Running Enhanced LLM to TTS Demo (Balanced Profile)...${NC}"
    cd "$SCRIPT_DIR"
    "$SCRIPT_DIR/.venv/bin/python" llm_to_tts_demo_enhanced.py
}

# Function to run interactive demo
run_interactive() {
    echo -e "\n${GREEN}Running Interactive LLM to TTS Demo...${NC}"
    cd "$SCRIPT_DIR"
    "$SCRIPT_DIR/.venv/bin/python" llm_to_tts_demo_enhanced.py --interactive
}

# Function to run with specific profile
run_profile() {
    local profile="$1"
    echo -e "\n${GREEN}Running Enhanced Demo with '${profile}' Profile...${NC}"
    cd "$SCRIPT_DIR"
    "$SCRIPT_DIR/.venv/bin/python" llm_to_tts_demo_enhanced.py --profile "$profile"
}

# Main script logic
case "${1:-default}" in
    "interactive")
        run_interactive
        ;;
    "fast"|"balanced"|"quality")
        run_profile "$1"
        ;;
    "help"|"-h"|"--help")
        show_usage
        ;;
    "default"|"")
        run_default
        ;;
    *)
        echo -e "${RED}Unknown option: $1${NC}"
        show_usage
        exit 1
        ;;
esac

echo -e "\n${GREEN}Demo completed!${NC}"
echo -e "Check the 'demo/output' directory for generated audio files."
echo -e "Logs are saved in 'demo/demo.log'"
echo -e "For more information, see: demo/README.md"
