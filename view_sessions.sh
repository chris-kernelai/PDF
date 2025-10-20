#!/bin/bash
################################################################################
# view_sessions.sh
#
# View pipeline session log in a formatted way
#
# Usage:
#   ./view_sessions.sh                    # Show all sessions
#   ./view_sessions.sh --latest 10        # Show latest 10 sessions
#   ./view_sessions.sh --session abc123   # Show specific session
#   ./view_sessions.sh --status FAILED    # Show only failed sessions
################################################################################

SESSION_LOG="pipeline_sessions.log"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

show_help() {
    echo "Pipeline Session Viewer"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --latest N        Show latest N sessions (default: all)"
    echo "  --session ID      Show specific session ID"
    echo "  --status STATUS   Filter by status (STARTED, COMPLETED, FAILED)"
    echo "  --help            Show this help"
    echo ""
    echo "Examples:"
    echo "  $0                           # Show all sessions"
    echo "  $0 --latest 5                # Show latest 5 sessions"
    echo "  $0 --session a1b2c3d4        # Show session a1b2c3d4"
    echo "  $0 --status FAILED           # Show only failed sessions"
}

# Parse arguments
LATEST=""
SESSION_ID=""
STATUS_FILTER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --latest)
            LATEST="$2"
            shift 2
            ;;
        --session)
            SESSION_ID="$2"
            shift 2
            ;;
        --status)
            STATUS_FILTER="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if log file exists
if [ ! -f "$SESSION_LOG" ]; then
    echo -e "${RED}❌ Session log file not found: $SESSION_LOG${NC}"
    echo "Run the pipeline first to generate session logs."
    exit 1
fi

# Build filter command
FILTER_CMD="cat $SESSION_LOG"

if [ -n "$SESSION_ID" ]; then
    FILTER_CMD="$FILTER_CMD | grep \"Session: $SESSION_ID\""
fi

if [ -n "$STATUS_FILTER" ]; then
    FILTER_CMD="$FILTER_CMD | grep \"Status: $STATUS_FILTER\""
fi

if [ -n "$LATEST" ]; then
    FILTER_CMD="$FILTER_CMD | tail -n $LATEST"
fi

# Execute filter and format output
eval "$FILTER_CMD" | while IFS='|' read -r timestamp session_info batch_info doc_range status_info; do
    # Clean up the fields
    timestamp=$(echo "$timestamp" | xargs)
    session_id=$(echo "$session_info" | sed 's/Session: //' | xargs)
    batch_num=$(echo "$batch_info" | sed 's/Batch: //' | xargs)
    doc_range=$(echo "$doc_range" | sed 's/Doc Range: //' | xargs)
    status=$(echo "$status_info" | sed 's/Status: //' | xargs)
    
    # Color code the status
    case "$status" in
        "STARTED")
            status_color="${YELLOW}▶${NC}"
            status_text="${YELLOW}STARTED${NC}"
            ;;
        "COMPLETED")
            status_color="${GREEN}✓${NC}"
            status_text="${GREEN}COMPLETED${NC}"
            ;;
        "FAILED")
            status_color="${RED}✗${NC}"
            status_text="${RED}FAILED${NC}"
            ;;
        *)
            status_color="${BLUE}?${NC}"
            status_text="${BLUE}$status${NC}"
            ;;
    esac
    
    # Format the output
    printf "${CYAN}%s${NC} | ${BLUE}Session: %s${NC} | ${CYAN}Batch: %s${NC} | ${CYAN}Docs: %s${NC} | %s %s\n" \
        "$timestamp" "$session_id" "$batch_num" "$doc_range" "$status_color" "$status_text"
done

# Show summary if no filters
if [ -z "$SESSION_ID" ] && [ -z "$STATUS_FILTER" ]; then
    echo ""
    echo -e "${BLUE}Summary:${NC}"
    total_sessions=$(wc -l < "$SESSION_LOG")
    completed_sessions=$(grep "Status: COMPLETED" "$SESSION_LOG" | wc -l)
    failed_sessions=$(grep "Status: FAILED" "$SESSION_LOG" | wc -l)
    started_sessions=$(grep "Status: STARTED" "$SESSION_LOG" | wc -l)
    
    echo "  Total sessions: $total_sessions"
    echo "  Completed: $completed_sessions"
    echo "  Failed: $failed_sessions"
    echo "  In progress: $started_sessions"
fi
