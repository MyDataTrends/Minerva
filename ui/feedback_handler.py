"""
User Feedback Handler for Assay.

Provides comprehensive feedback collection:
- Thumbs up/down buttons after chat responses
- Typed feedback parsing (detects "that's wrong", "perfect!", etc.)
- "Report Issue" button capturing session state + logs
- Integration with InteractionLogger for learning

Usage:
    from ui.feedback_handler import render_feedback_buttons, parse_typed_feedback, render_report_issue
    
    # After displaying assistant response
    render_feedback_buttons(interaction_id, response_content)
    
    # In chat input processing
    feedback = parse_typed_feedback(user_message)
    if feedback:
        handle_feedback(feedback)
"""
import streamlit as st
import logging
import re
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Import diagnostics
try:
    from ui.diagnostics import get_diagnostics, capture_error
    DIAGNOSTICS_AVAILABLE = True
except ImportError:
    DIAGNOSTICS_AVAILABLE = False

# Import interaction logger
try:
    from llm_learning.interaction_logger import get_interaction_logger
    LOGGER_AVAILABLE = True
except ImportError:
    LOGGER_AVAILABLE = False


@dataclass
class FeedbackResult:
    """Result of feedback parsing or button interaction."""
    feedback_type: str  # "positive", "negative", "correction", "clarification", "report"
    confidence: float  # 0-1 confidence in detection
    details: str  # Additional details or user comment
    interaction_id: Optional[int] = None  # Associated interaction if known
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_type": self.feedback_type,
            "confidence": self.confidence,
            "details": self.details,
            "interaction_id": self.interaction_id,
        }


# Feedback pattern matchers
POSITIVE_PATTERNS = [
    (r"(?i)\b(perfect|excellent|great|awesome|nice|good job|well done|thanks?|thank you|correct|right|exactly)\b", 0.8),
    (r"(?i)^(yes|yep|yeah|yup|ok|okay)$", 0.6),
    (r"(?i)\b(that'?s? (?:what i (?:wanted|needed)|right|correct|perfect))\b", 0.9),
    (r"👍|👏|🎉|✅|💯", 0.95),
]

NEGATIVE_PATTERNS = [
    (r"(?i)\b(wrong|incorrect|not right|no that'?s? (?:not|wrong)|bad|error|mistake)\b", 0.85),
    (r"(?i)\b(that'?s? not what i (?:meant|wanted|asked))\b", 0.9),
    (r"(?i)^(no|nope|nah)$", 0.6),
    (r"(?i)\b(doesn'?t work|didn'?t work|broken|failed)\b", 0.8),
    (r"👎|❌|🚫", 0.95),
]

CORRECTION_PATTERNS = [
    (r"(?i)\b(actually|instead|should be|it should|i meant|what i meant)\b", 0.7),
    (r"(?i)\b(change|fix|correct|update|modify) (?:it|this|that) to\b", 0.85),
    (r"(?i)\b(not.*but|rather than)\b", 0.7),
]

CLARIFICATION_PATTERNS = [
    (r"(?i)\b(what do you mean|i don'?t understand|can you explain|clarify)\b", 0.8),
    (r"(?i)^\?\s*$", 0.6),
    (r"(?i)\b(confused|unclear|what)\?$", 0.7),
]


def parse_typed_feedback(message: str) -> Optional[FeedbackResult]:
    """
    Parse typed user message for implicit feedback.
    
    Args:
        message: The user's typed message
        
    Returns:
        FeedbackResult if feedback detected, None otherwise
    """
    message = message.strip()
    if not message:
        return None
    
    # Check each pattern category
    for pattern, confidence in POSITIVE_PATTERNS:
        if re.search(pattern, message):
            return FeedbackResult(
                feedback_type="positive",
                confidence=confidence,
                details=message,
            )
    
    for pattern, confidence in NEGATIVE_PATTERNS:
        if re.search(pattern, message):
            return FeedbackResult(
                feedback_type="negative",
                confidence=confidence,
                details=message,
            )
    
    for pattern, confidence in CORRECTION_PATTERNS:
        if re.search(pattern, message):
            return FeedbackResult(
                feedback_type="correction",
                confidence=confidence,
                details=message,
            )
    
    for pattern, confidence in CLARIFICATION_PATTERNS:
        if re.search(pattern, message):
            return FeedbackResult(
                feedback_type="clarification",
                confidence=confidence,
                details=message,
            )
    
    return None


def handle_feedback(
    feedback: FeedbackResult,
    interaction_id: Optional[int] = None,
) -> None:
    """
    Process feedback and update the interaction logger.
    
    Args:
        feedback: The parsed or button-generated feedback
        interaction_id: ID of the interaction being rated
    """
    if not LOGGER_AVAILABLE:
        logger.warning("InteractionLogger not available for feedback")
        return
    
    try:
        il = get_interaction_logger()
        
        if interaction_id:
            if feedback.feedback_type == "positive":
                il.update_rating(interaction_id, 1.0)
                logger.info(f"Recorded positive feedback for interaction {interaction_id}")
            elif feedback.feedback_type == "negative":
                il.update_rating(interaction_id, 0.0)
                logger.info(f"Recorded negative feedback for interaction {interaction_id}")
            elif feedback.feedback_type == "correction":
                il.mark_corrected(interaction_id)
                logger.info(f"Marked interaction {interaction_id} as corrected")
        
        # Log to diagnostics if available
        if DIAGNOSTICS_AVAILABLE:
            diag = get_diagnostics()
            diag.log_info(
                "User feedback",
                f"{feedback.feedback_type}: {feedback.details[:100]}",
                context={"interaction_id": interaction_id, "confidence": feedback.confidence}
            )
    
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}")


def render_feedback_buttons(
    interaction_id: int,
    response_content: str,
    key_prefix: str = "",
) -> Optional[FeedbackResult]:
    """
    Render thumbs up/down feedback buttons after a response.
    
    Args:
        interaction_id: ID of the logged interaction
        response_content: The assistant's response (for context)
        key_prefix: Unique prefix for button keys
        
    Returns:
        FeedbackResult if user clicked a button, None otherwise
    """
    # Create unique keys
    up_key = f"{key_prefix}_thumbs_up_{interaction_id}"
    down_key = f"{key_prefix}_thumbs_down_{interaction_id}"
    feedback_key = f"{key_prefix}_feedback_given_{interaction_id}"
    
    # Check if feedback already given
    if st.session_state.get(feedback_key):
        st.caption("✓ Feedback recorded")
        return None
    
    col1, col2, col3 = st.columns([1, 1, 10])
    
    result = None
    
    with col1:
        if st.button("👍", key=up_key, help="This was helpful"):
            result = FeedbackResult(
                feedback_type="positive",
                confidence=1.0,
                details="Thumbs up button clicked",
                interaction_id=interaction_id,
            )
            st.session_state[feedback_key] = True
            handle_feedback(result, interaction_id)
            st.toast("Thanks for the feedback! 👍")
    
    with col2:
        if st.button("👎", key=down_key, help="This wasn't helpful"):
            result = FeedbackResult(
                feedback_type="negative",
                confidence=1.0,
                details="Thumbs down button clicked",
                interaction_id=interaction_id,
            )
            st.session_state[feedback_key] = True
            handle_feedback(result, interaction_id)
            st.toast("Thanks for the feedback. We'll improve! 📝")
    
    return result


def render_feedback_with_comment(
    interaction_id: int,
    key_prefix: str = "",
) -> Optional[FeedbackResult]:
    """
    Render feedback buttons with optional comment field.
    """
    feedback_key = f"{key_prefix}_feedback_given_{interaction_id}"
    
    if st.session_state.get(feedback_key):
        st.caption("✓ Feedback recorded")
        return None
    
    with st.expander("💬 Provide Feedback", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            rating = st.radio(
                "How was this response?",
                ["👍 Helpful", "👎 Not helpful", "🔄 Partially correct"],
                key=f"{key_prefix}_rating_{interaction_id}",
                horizontal=True,
            )
        
        comment = st.text_input(
            "Additional comments (optional)",
            key=f"{key_prefix}_comment_{interaction_id}",
            placeholder="What could be improved?"
        )
        
        if st.button("Submit Feedback", key=f"{key_prefix}_submit_{interaction_id}"):
            feedback_type = {
                "👍 Helpful": "positive",
                "👎 Not helpful": "negative",
                "🔄 Partially correct": "correction",
            }.get(rating, "positive")
            
            result = FeedbackResult(
                feedback_type=feedback_type,
                confidence=1.0,
                details=comment or rating,
                interaction_id=interaction_id,
            )
            
            st.session_state[feedback_key] = True
            handle_feedback(result, interaction_id)
            st.success("Feedback recorded! Thank you.")
            return result
    
    return None


def render_report_issue():
    """
    Render a "Report Issue" button that captures session state and logs.
    """
    with st.expander("🐛 Report an Issue", expanded=False):
        st.write("Capture current session state and logs for debugging.")
        
        issue_type = st.selectbox(
            "Issue Type",
            ["Bug", "Incorrect Response", "Crash/Error", "Feature Request", "Other"],
        )
        
        description = st.text_area(
            "Describe the issue",
            placeholder="What happened? What did you expect?",
            height=100,
        )
        
        include_chat = st.checkbox("Include chat history", value=True)
        include_logs = st.checkbox("Include recent error logs", value=True)
        
        if st.button("📤 Generate Report"):
            report = _generate_issue_report(
                issue_type=issue_type,
                description=description,
                include_chat=include_chat,
                include_logs=include_logs,
            )
            
            # Offer download
            st.download_button(
                "📥 Download Report",
                report,
                file_name=f"assay_issue_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
            )
            
            st.success("Report generated! Download and share with the development team.")


def _generate_issue_report(
    issue_type: str,
    description: str,
    include_chat: bool = True,
    include_logs: bool = True,
) -> str:
    """Generate a comprehensive issue report."""
    lines = [
        "# Assay Issue Report",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Issue Type:** {issue_type}",
        "",
        "## Description",
        description or "No description provided.",
        "",
    ]
    
    # Session info
    lines.append("## Session Info")
    try:
        from ui.session_context import get_context
        ctx = get_context()
        stats = ctx.get_memory_stats()
        lines.append(f"- Chat messages: {stats.get('chat_messages', 0)}")
        lines.append(f"- Datasets loaded: {stats.get('datasets_loaded', 0)}")
        lines.append(f"- Actions logged: {stats.get('action_log_entries', 0)}")
    except Exception as e:
        lines.append(f"- Could not get session stats: {e}")
    lines.append("")
    
    # Chat history
    if include_chat:
        lines.append("## Recent Chat History")
        try:
            from ui.session_context import get_context
            ctx = get_context()
            recent = ctx.get_recent_messages(10)
            for msg in recent:
                role = msg.get("role", "unknown").upper()
                content = msg.get("content", "")[:500]
                lines.append(f"**{role}:** {content}")
                lines.append("")
        except Exception as e:
            lines.append(f"Could not get chat history: {e}")
        lines.append("")
    
    # Error logs
    if include_logs:
        lines.append("## Recent Errors")
        try:
            if DIAGNOSTICS_AVAILABLE:
                diag = get_diagnostics()
                errors = diag.get_recent_errors(5)
                for err in errors:
                    lines.append(f"### {err.operation}")
                    lines.append(f"- **Time:** {err.timestamp}")
                    lines.append(f"- **Error:** {err.error_type}: {err.error_detail}")
                    if err.traceback:
                        lines.append("```")
                        lines.append(err.traceback[:500])
                        lines.append("```")
                    lines.append("")
            else:
                lines.append("Diagnostics not available.")
        except Exception as e:
            lines.append(f"Could not get error logs: {e}")
    
    return "\n".join(lines)


# =============================================================================
# Integration Helpers
# =============================================================================

def get_last_interaction_id() -> Optional[int]:
    """Get the ID of the most recent logged interaction."""
    if not LOGGER_AVAILABLE:
        return None
    try:
        il = get_interaction_logger()
        recent = il.get_recent_interactions(limit=1)
        if recent:
            return recent[0].get("id")
    except Exception:
        pass
    return None


def check_and_handle_typed_feedback(message: str) -> Tuple[bool, Optional[FeedbackResult]]:
    """
    Check if a message contains feedback and handle it.
    
    Returns:
        Tuple of (is_feedback, feedback_result)
        If is_feedback is True, the message should not be processed as a query.
    """
    # Only treat as pure feedback if it's short and matches patterns with high confidence
    feedback = parse_typed_feedback(message)
    
    if feedback and feedback.confidence >= 0.8 and len(message) < 50:
        # This looks like pure feedback, not a query
        interaction_id = get_last_interaction_id()
        handle_feedback(feedback, interaction_id)
        return True, feedback
    
    # Even if there's feedback, if the message is longer it's probably a new query
    return False, feedback
