#!/usr/bin/env python3
"""
Documentation Analyzer for Minerva

Analyzes project documentation and generates a vision alignment report.
"""
import argparse
from pathlib import Path
from typing import Dict, List
import re

def analyze_documentation(docs_dir: str = "docs") -> Dict:
    """Analyze project documentation and generate insights."""
    analysis = {
        "documents_found": [],
        "key_components": set(),
        "use_cases": [],
        "architecture_elements": set(),
        "missing_elements": []
    }
    
    docs_path = Path(docs_dir)
    
    # Analyze each markdown file
    for doc_path in docs_path.glob("*.md"):
        analysis["documents_found"].append(doc_path.name)
        content = doc_path.read_text(encoding='utf-8')
        
        # Simple pattern matching (can be enhanced with NLP)
        if "architecture" in doc_path.name.lower():
            analysis["architecture_elements"].update(
                re.findall(r"##\s+(.*?)\n", content, re.IGNORECASE)
            )
        elif "use_case" in doc_path.name.lower() or "usecase" in doc_path.name.lower():
            cases = re.findall(r"###\s+(.*?)\n", content, re.IGNORECASE)
            analysis["use_cases"].extend(cases)
    
    # Add README analysis
    readme = Path("README.md")
    if readme.exists():
        analysis["documents_found"].append("README.md")
        readme_content = readme.read_text(encoding='utf-8')
        analysis["key_components"].update(
            re.findall(r"`([a-zA-Z0-9_/]+)`", readme_content)
        )
    
    return analysis

def generate_report(analysis: Dict, output_file: str) -> None:
    """Generate a markdown report from the analysis."""
    report = [
        "# Minerva Vision and Documentation Analysis",
        "\n## Documentation Overview",
        f"Found {len(analysis['documents_found'])} documentation files.",
        "\n## Key Components Identified",
        *[f"- {comp}" for comp in sorted(analysis["key_components"])],
        "\n## Architecture Elements",
        *[f"- {elem}" for elem in sorted(analysis["architecture_elements"])],
        "\n## Use Cases",
        *[f"- {uc}" for uc in analysis["use_cases"]],
        "\n## Recommendations",
        "1. **Documentation Coverage**",
        "   - Consider adding missing READMEs for key modules",
        "   - Add architecture decision records (ADRs) for major decisions",
        "\n2. **Code-Docs Alignment**",
        "   - Verify all documented components exist in code",
        "   - Ensure all major code components are documented",
        "\n3. **Vision Alignment**",
        "   - Review original project goals vs current implementation",
        "   - Identify any technical debt or architectural drift"
    ]
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(report))

def main():
    parser = argparse.ArgumentParser(description='Analyze project documentation.')
    parser.add_argument('--output', default='vision_report.md', 
                       help='Output file for the analysis report')
    args = parser.parse_args()
    
    print("Analyzing documentation...")
    analysis = analyze_documentation()
    generate_report(analysis, args.output)
    print(f"Report generated: {args.output}")

if __name__ == "__main__":
    main()