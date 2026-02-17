# Escalation Policy

## Priority Levels

| Level | Label | Response Time | Action |
|:---|:---|:---|:---|
| 🔴 Urgent | `urgent` | Same day | Requires human decision |
| 🟡 Review | `review` | Within 48 hours | Needs human review |
| 🟢 FYI | `fyi` | No action needed | Handled autonomously |
| 📊 Metric | `metric` | Weekly review | Data point for analysis |

## Escalation Rules

1. **Security issues** are always 🔴 regardless of source
2. **Bugs** start as 🟡 unless they affect core data pipeline (then 🔴)
3. **Feature requests** are 🟢 (acknowledged, tracked)
4. **Questions** are 🟢 (auto-responded by Advocate)
5. **Engineer PRs** scoring <7 are 🟢 FYI (sent back for iteration)
6. **Engineer PRs** scoring ≥7 are 🟡 (ready for human review)
7. **Spending** above $50/month threshold is 🔴
