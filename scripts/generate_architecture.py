#!/usr/bin/env python3
"""
Architecture Visualization Generator for Minerva

Generates interactive HTML and Markdown visualizations of the project architecture.
"""
import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple
import argparse
import json

class ArchitectureAnalyzer:
    def __init__(self, root_dir: str = "."):
        self.root_dir = Path(root_dir)
        self.modules: Dict[str, Dict] = {}
        self.imports: Dict[str, Set[str]] = {}
        self.endpoints: List[Dict] = []
        self.data_flows: List[Tuple[str, str, str]] = []

    def analyze_python_file(self, file_path: Path):
        """Analyze a Python file for imports and FastAPI endpoints."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse the Python file
            tree = ast.parse(content, filename=str(file_path))
            
            # Get relative import path
            rel_path = str(file_path.relative_to(self.root_dir))
            self.modules[rel_path] = {
                'type': 'module',
                'imports': set(),
                'endpoints': []
            }
            
            # Find imports and API endpoints
            for node in ast.walk(tree):
                # Find imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    for name in node.names:
                        if isinstance(node, ast.Import):
                            self.modules[rel_path]['imports'].add(name.name.split('.')[0])
                        else:
                            self.modules[rel_path]['imports'].add(node.module.split('.')[0] if node.module else '')
                
                # Find FastAPI endpoints
                if (isinstance(node, ast.FunctionDef) and 
                    any(d.id == 'app' for d in node.decorator_list if hasattr(d, 'id'))):
                    for decorator in node.decorator_list:
                        if hasattr(decorator, 'func') and hasattr(decorator.func, 'attr') and decorator.func.attr in ['get', 'post', 'put', 'delete']:
                            self.modules[rel_path]['endpoints'].append({
                                'method': decorator.func.attr.upper(),
                                'path': decorator.args[0].s if decorator.args else '/',
                                'function': node.name
                            })
                            self.endpoints.append({
                                'file': rel_path,
                                'method': decorator.func.attr.upper(),
                                'path': decorator.args[0].s if decorator.args else '/',
                                'function': node.name
                            })
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")

    def analyze_directory(self):
        """Analyze all Python files in the project."""
        for root, _, files in os.walk(self.root_dir):
            # Skip virtual environment and other non-relevant directories
            if any(skip in root for skip in ['venv', '.git', '__pycache__', '.pytest_cache']):
                continue
                
            for file in files:
                if file.endswith('.py'):
                    self.analyze_python_file(Path(root) / file)

    def generate_markdown(self, output_file: str):
        """Generate markdown documentation."""
        md = ["# Minerva Architecture Overview\n"]
        
        # Modules section
        md.append("## Modules\n")
        for module, data in self.modules.items():
            md.append(f"### {module}")
            if data['imports']:
                md.append("#### Imports")
                md.extend([f"- {imp}" for imp in sorted(data['imports'])])
            if data['endpoints']:
                md.append("#### Endpoints")
                for ep in data['endpoints']:
                    md.append(f"- `{ep['method']} {ep['path']}` - {ep['function']}")
            md.append("")
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(md))
        
        print(f"Markdown documentation generated: {output_file}")

    def generate_interactive_html(self, output_file: str):
        """Generate interactive HTML visualization."""
        # Prepare nodes and links for the graph
        nodes = []
        links = []
        
        # Add modules as nodes
        for i, module in enumerate(self.modules.keys()):
            nodes.append({
                'id': module,
                'name': module.split('/')[-1],
                'group': module.split('/')[0] if '/' in module else 'root',
                'type': 'module'
            })
            
            # Add imports as links
            for imp in self.modules[module]['imports']:
                if imp in [m.split('/')[-1].replace('.py', '') for m in self.modules.keys()]:
                    target = next((m for m in self.modules.keys() 
                                 if m.split('/')[-1].replace('.py', '') == imp), None)
                    if target:
                        links.append({
                            'source': module,
                            'target': target,
                            'value': 1
                        })
        
        # Add endpoints as nodes and links
        for i, ep in enumerate(self.endpoints, len(nodes)):
            node_id = f"endpoint_{i}"
            nodes.append({
                'id': node_id,
                'name': f"{ep['method']} {ep['path']}",
                'group': 'endpoint',
                'type': 'endpoint'
            })
            links.append({
                'source': ep['file'],
                'target': node_id,
                'value': 2
            })
        
        # Generate HTML with D3.js visualization
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Minerva Architecture</title>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ margin: 0; overflow: hidden; }}
        #graph {{ width: 100%; height: 100vh; }}
        .node {{ stroke: #fff; stroke-width: 1.5px; }}
        .link {{ stroke: #999; stroke-opacity: 0.6; }}
        .module {{ fill: #1f77b4; }}
        .endpoint {{ fill: #ff7f0e; }}
        .label {{ font: 10px sans-serif; pointer-events: none; }}
    </style>
</head>
<body>
    <div id="graph"></div>
    <script>
        const data = {json.dumps({'nodes': nodes, 'links': links}, indent=8)};
        
        const width = window.innerWidth;
        const height = window.innerHeight;
        
        const color = d3.scaleOrdinal(d3.schemeCategory10);
        
        const simulation = d3.forceSimulation(data.nodes)
            .force("link", d3.forceLink(data.links).id(d => d.id).distance(100))
            .force("charge", d3.forceManyBody().strength(-500))
            .force("center", d3.forceCenter(width / 2, height / 2))
            .force("collision", d3.forceCollide().radius(40));
            
        const svg = d3.select("#graph")
            .append("svg")
            .attr("width", "100%")
            .attr("height", "100%");
            
        const link = svg.append("g")
            .selectAll("line")
            .data(data.links)
            .enter().append("line")
            .attr("class", "link");
            
        const node = svg.append("g")
            .selectAll(".node")
            .data(data.nodes)
            .enter().append("g")
            .call(d3.drag()
                .on("start", dragstarted)
                .on("drag", dragged)
                .on("end", dragended));
                
        node.append("circle")
            .attr("r", 8)
            .attr("class", d => d.type);
            
        node.append("text")
            .attr("dx", 12)
            .attr("dy", ".35em")
            .text(d => d.name)
            .attr("class", "label");
            
        simulation.on("tick", () => {{
            link
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
                
            node.attr("transform", "function(d) {{ return 'translate(' + d.x + ',' + d.y + ')'; }}");
        }});
        
        function dragstarted(event, d) {{
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }}
        
        function dragged(event, d) {{
            d.fx = event.x;
            d.fy = event.y;
        }}
        
        function dragended(event, d) {{
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }}
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            const width = window.innerWidth;
            const height = window.innerHeight;
            
            d3.select("svg")
                .attr("width", width)
                .attr("height", height);
                
            simulation.force("center", d3.forceCenter(width / 2, height / 2));
            simulation.alpha(1).restart();
        }});
    </script>
</body>
</html>
"""
        
        # Write HTML file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html)
            
        print(f"Interactive HTML visualization generated: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate architecture documentation.')
    parser.add_argument('--root', default='.', help='Root directory of the project')
    parser.add_argument('--markdown', default='ARCHITECTURE.md', help='Output markdown file')
    parser.add_argument('--html', default='architecture.html', help='Output HTML file')
    args = parser.parse_args()
    
    print("Analyzing project structure...")
    analyzer = ArchitectureAnalyzer(args.root)
    analyzer.analyze_directory()
    
    print("Generating documentation...")
    analyzer.generate_markdown(args.markdown)
    analyzer.generate_interactive_html(args.html)
    
    print("\nDone! Open the HTML file in a browser to explore the interactive visualization.")

if __name__ == "__main__":
    main()