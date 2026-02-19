"""
Data Fabric Tab - Visualizing Semantic Connectivity.

This module renders the "Data Fabric" tab, showing users how their data
connects to the broader world of public knowledge (World Bank, FRED, etc.).
It builds trust by surfacing the "Magic" of automatic enrichment.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import networkx as nx

def render_data_fabric_tab(df: pd.DataFrame, meta: list = None):
    """
    Render the main Data Fabric view.
    """
    st.header("🕸️ Data Fabric & Lineage")
    st.markdown("""
    Assay doesn't just see your data; it understands it. 
    Here is how your local file connects to global knowledge.
    """)

    if df is None:
        st.info("Upload a dataset to see its connections.")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🔗 Semantic Graph")
        # Placeholder for a beautiful node-link diagram
        # showing [Your Data] --(ISO Code)--> [World Bank]
        render_lineage_graph(df, meta)

    with col2:
        st.subheader("🧩 Detected Entities")
        render_entity_list(df, meta)

    st.divider()
    
    # Auto-Discovery Section
    render_auto_discovery_section()
    
    st.divider()
    
    st.subheader("🌍 Enrichment Opportunities")
    render_enrichment_options(df)


def render_lineage_graph(df: pd.DataFrame, meta: list):
    """
    Render a NetworkX/Plotly graph showing data relationships.
    """
    # Create graph
    G = nx.Graph()
    
    # Add User Data Node
    G.add_node("Your Dataset", type="source", color="#2563eb", icon="📄")
    
    # Detect entities from actual columns
    for col in df.columns:
        col_lower = col.lower()
        semantic_type = None
        
        # Check dtype and column names
        if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col_lower:
            semantic_type = "date"
        elif col_lower in ["country", "country_code", "iso", "iso2", "iso3"]:
            semantic_type = "country_code"
        elif "gdp" in col_lower or "gnp" in col_lower:
            semantic_type = "economic_indicator"
        elif "rate" in col_lower:
            semantic_type = "rate"
            
        if semantic_type:
            node_name = f"{col} ({semantic_type})"
            G.add_node(node_name, type="column", color="#10b981", icon="🏷️")
            G.add_edge("Your Dataset", node_name)
            
            # Link to External Sources based on type
            if semantic_type in ['country_code']:
                G.add_node("World Bank Data", type="external", color="#f59e0b", icon="🌍")
                G.add_edge(node_name, "World Bank Data")
            
            if semantic_type in ['date', 'economic_indicator', 'rate']:
                G.add_node("FRED Economic", type="external", color="#ef4444", icon="📈")
                G.add_edge(node_name, "FRED Economic")

    # Layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw edges
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines')

    # Draw nodes
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        
        # Color based on type
        ntype = G.nodes[node].get("type", "default")
        if ntype == "source": node_color.append("#2563eb") # Blue
        elif ntype == "column": node_color.append("#10b981") # Green
        elif ntype == "external": node_color.append("#f59e0b") # Orange
        else: node_color.append("#94a3b8") # Grey

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        textposition="top center",
        marker=dict(
            showscale=False,
            color=node_color,
            size=20,
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=0,l=0,r=0,t=0),
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )
    
    st.plotly_chart(fig, use_container_width=True)


def render_entity_list(df: pd.DataFrame, meta: list):
    """
    Render a list of detected entities by analyzing the actual data.
    """
    if df is None:
        st.caption("No data to analyze.")
        return

    found_entities = []
    
    for col in df.columns:
        semantic_type = None
        col_lower = col.lower()
        
        # Check dtype first
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            semantic_type = "date"
        elif "date" in col_lower or "time" in col_lower:
            semantic_type = "date"
        elif col_lower in ["country", "country_code", "iso", "iso2", "iso3", "nation"]:
            semantic_type = "country_code"
        elif col_lower in ["state", "us_state", "province"]:
            semantic_type = "us_state"
        elif col_lower in ["zip", "zipcode", "postal"]:
            semantic_type = "zip_code"
        elif "gdp" in col_lower or "gnp" in col_lower:
            semantic_type = "economic_indicator"
        elif "rate" in col_lower or "percent" in col_lower:
            semantic_type = "rate_percent"
            
        if semantic_type:
            found_entities.append({"column": col, "type": semantic_type})

    if not found_entities:
        st.info("No semantic entities (like countries or dates) found in this dataset.")
        return

    for entity in found_entities:
        with st.expander(f"📌 {entity['column']}", expanded=True):
            st.write(f"**Identified as:** `{entity['type']}`")
            if entity['type'] == "date":
                st.success("✅ Ready for time-series analysis")
            elif entity['type'] in ["country_code", "us_state"]:
                st.success("✅ Ready for geographic linking")
            else:
                st.success("✅ Ready for enrichment")


def render_auto_discovery_section():
    """
    Render the Auto-Discovery controls for automatic API access.
    
    Provides:
    1. Toggle to enable autonomous data fetching
    2. Text input for describing data needs
    3. Cascade-style prompts showing discovered APIs before fetching
    """
    # Use unified session context
    from ui.session_context import get_context, log_tab_action
    ctx = get_context()
    
    st.subheader("🔮 Smart Data Discovery")
    
    # Auto-mode toggle uses preference system
    col_toggle, col_description = st.columns([1, 3])
    
    with col_toggle:
        auto_mode = st.toggle(
            "🚀 Auto-Discovery",
            value=ctx.preferences.get("auto_discovery", False),
            key="auto_discovery_toggle",
            help="Allow Assay to automatically find and suggest external data sources"
        )
        ctx.set_preference("auto_discovery", auto_mode)
    
    with col_description:
        if auto_mode:
            st.success("**Auto-Discovery Enabled** — Assay will suggest relevant APIs for your queries")
        else:
            st.info("Enable to let Assay find external data sources that match your needs")
    
    if not auto_mode:
        return
    
    # Query input
    st.markdown("**Describe the data you need:**")
    query = st.text_input(
        "Data Query",
        placeholder="e.g., 'US unemployment trends over the last decade' or 'Compare GDP growth across European countries'",
        key="discovery_query_input",
        label_visibility="collapsed"
    )
    
    # Action buttons - now with Auto-Fetch option
    col_search, col_autofetch, col_clear = st.columns([1, 1, 2])
    with col_search:
        search_clicked = st.button("🔍 Find Sources", help="Find matching APIs without fetching")
    with col_autofetch:
        auto_fetch_clicked = st.button(
            "⚡ Auto-Fetch", 
            type="primary",
            help="Automatically find, connect, and fetch data",
            disabled=not query
        )
    with col_clear:
        if st.button("Clear"):
            ctx.discoveries = []
            ctx.pending_fetch = None
            if "auto_fetch_result" in st.session_state:
                del st.session_state["auto_fetch_result"]
            st.rerun()
    
    # AUTO-FETCH: One-click autonomous flow
    if auto_fetch_clicked and query:
        with st.spinner("🔮 Discovering APIs, generating connector, and fetching data..."):
            try:
                from mcp_server.discovery_agent import get_discovery_agent
                import pandas as pd
                import logging
                
                # Pass session master password for Kaggle access
                master_pw = st.session_state.get("kaggle_master_password")
                
                # Check if we have an API key for retry
                retry_api_key = st.session_state.get("pending_api_key")
                if retry_api_key:
                    del st.session_state["pending_api_key"]
                
                agent = get_discovery_agent(master_password=master_pw)
                result = agent.one_click_fetch_rich(query, api_key=retry_api_key)
                
                # Store structured result for UI
                st.session_state["auto_fetch_result"] = {
                    "success": result.success,
                    "data": result.data,
                    "status": result.status,
                    "query": query,
                    # Auth info for guided UI
                    "needs_auth": result.needs_auth,
                    "api_name": result.api_name,
                    "api_id": result.api_id,
                    "auth_type": result.auth_type,
                    "signup_url": result.signup_url,
                    "auth_instructions": result.auth_instructions
                }
                
            except Exception as e:
                st.error(f"Auto-fetch failed: {e}")
    
    # Display auto-fetch results
    if "auto_fetch_result" in st.session_state:
        result = st.session_state["auto_fetch_result"]
        
        # Check if API needs authentication
        if result.get("needs_auth"):
            st.warning(f"🔐 **{result.get('api_name', 'This API')}** requires authentication")
            
            # Auth info container
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Authentication Type:** `{result.get('auth_type', 'api_key')}`")
                    
                    if result.get("auth_instructions"):
                        st.caption(result["auth_instructions"])
                    
                    # Signup URL
                    if result.get("signup_url"):
                        st.markdown(f"**Get your API key:** [{result['signup_url']}]({result['signup_url']})")
                    
                    # API Key input
                    api_key_input = st.text_input(
                        "Enter API Key",
                        type="password",
                        key=f"auth_key_{result.get('api_id', 'api')}",
                        help="Paste your API key here"
                    )
                    
                    # Option to save for future
                    save_key = st.checkbox(
                        "Save this key for future use",
                        value=True,
                        key="save_api_key_checkbox",
                        help="Stores encrypted in ~/.assay/credentials.json"
                    )
                
                # Action buttons
                col_retry, col_skip, col_cancel = st.columns(3)
                
                with col_retry:
                    if st.button("🔑 Connect with Key", disabled=not api_key_input, type="primary"):
                        # Save key if requested
                        if save_key and api_key_input:
                            try:
                                from mcp_server.credential_manager import CredentialManager
                                master_pw = st.session_state.get("kaggle_master_password")
                                if master_pw:
                                    cred_mgr = CredentialManager()
                                    cred_mgr.store_credential(
                                        api_id=result.get("api_id", result.get("api_name", "unknown").lower()),
                                        api_key=api_key_input,
                                        master_password=master_pw
                                    )
                                    st.success("Saved API key!")
                            except Exception as e:
                                st.warning(f"Could not save key: {e}")
                        
                        # Retry with API key
                        st.session_state["pending_api_key"] = api_key_input
                        del st.session_state["auto_fetch_result"]
                        st.rerun()
                
                with col_skip:
                    if st.button("⏭️ Try Next Source"):
                        # Clear and try again (would need to skip this source)
                        del st.session_state["auto_fetch_result"]
                        st.info("Skipped - try a different query or configure auth in Settings")
                        st.rerun()
                
                with col_cancel:
                    if st.button("❌ Cancel"):
                        del st.session_state["auto_fetch_result"]
                        st.rerun()
        
        elif result.get("data"):
            # Success - show data
            st.success(f"**Result:** {result['status']}")
            
            import pandas as pd
            if isinstance(result["data"], dict):
                df = pd.DataFrame(result["data"])
            else:
                df = result["data"]
            st.dataframe(df, width="stretch")
            
            # Option to add to datasets
            if st.button("➕ Add to My Datasets"):
                dataset_name = f"auto_fetched_{result.get('query', 'data')[:20].replace(' ', '_')}"
                ctx.add_dataset(dataset_name, df, set_as_primary=True)
                st.success(f"Added as '{dataset_name}'")
                del st.session_state["auto_fetch_result"]
                st.rerun()
        
        else:
            # Failed - show status
            st.info(f"**Result:** {result['status']}")
    
    # Search for APIs
    if search_clicked and query:
        with st.spinner("Searching for relevant data sources..."):
            try:
                from mcp_server.semantic_router import semantic_search_apis
                results = semantic_search_apis(query, top_k=5)
                ctx.discoveries = results
                ctx.set_preference("last_discovery_query", query)
                log_tab_action("Data Fabric", "search", f"Searched for: {query}")
            except Exception as e:
                st.error(f"Search failed: {e}")
                # Fallback to keyword search
                try:
                    from mcp_server.api_registry import search_apis_by_query, get_api
                    from mcp_server.semantic_router import APIMatch
                    keyword_results = search_apis_by_query(query)
                    results = []
                    for m in keyword_results[:5]:
                        api = get_api(m["api_id"])
                        if api:
                            results.append(APIMatch(
                                api_id=m["api_id"],
                                name=api.name,
                                description=api.description,
                                score=m["score"] / 50.0,
                                confidence="medium",
                                matched_via="keyword"
                            ))
                    ctx.discoveries = results
                    ctx.set_preference("last_discovery_query", query)
                except Exception as e2:
                    st.error(f"Fallback search also failed: {e2}")
    
    # Display discovered APIs with cascade-style prompts
    discovered = ctx.discoveries
    
    if discovered:
        st.markdown("---")
        st.markdown(f"**Found {len(discovered)} potential data sources for:** _{ctx.preferences.get('last_discovery_query', '')}_")
        
        for i, api in enumerate(discovered):
            confidence_icon = {
                "high": "🟢",
                "medium": "🟡", 
                "low": "🟠"
            }.get(api.confidence, "⚪")
            
            with st.container():
                col_info, col_action = st.columns([3, 1])
                
                with col_info:
                    st.markdown(f"""
                    {confidence_icon} **{api.name}**  
                    {api.description}  
                    *Match: {api.score:.0%} ({api.matched_via})*
                    """)
                
                with col_action:
                    # Auto-Connect button - generates connector automatically
                    if st.button(f"⚡ Connect", key=f"autoconnect_{api.api_id}", type="primary"):
                        st.session_state["auto_connect_target"] = api.api_id
                        st.rerun()
                
                st.markdown("---")
        
        # Handle auto-connect request
        if "auto_connect_target" in st.session_state:
            target_api_id = st.session_state["auto_connect_target"]
            
            with st.spinner(f"🔧 Generating connector for {target_api_id}..."):
                try:
                    from mcp_server.api_registry import get_api
                    from mcp_server.discovery_agent import get_discovery_agent, DiscoveredAPI
                    
                    api_def = get_api(target_api_id)
                    if api_def:
                        # Create DiscoveredAPI from registry entry
                        discovered = DiscoveredAPI(
                            name=api_def.name,
                            description=api_def.description,
                            base_url=api_def.base_url,
                            docs_url=api_def.docs_url,
                            openapi_url=getattr(api_def, 'openapi_url', None),
                            auth_type=api_def.auth_type,
                            signup_url=api_def.signup_url,
                            source="registry"
                        )
                        
                        agent = get_discovery_agent()
                        result = agent.auto_connect(discovered)
                        
                        if result.success:
                            st.success(f"✅ Connector generated for **{result.api_name}**")
                            
                            if result.needs_auth:
                                st.warning(f"""
                                **🔑 API Key Required**  
                                {result.auth_instructions}  
                                [Sign up here]({result.signup_url})
                                """)
                            
                            with st.expander("View Connector Code"):
                                st.code(result.connector_code, language="python")
                            
                            if result.sample_data:
                                st.markdown("**Sample Data:**")
                                import pandas as pd
                                st.dataframe(pd.DataFrame(result.sample_data))
                        else:
                            st.error(f"❌ {result.error}")
                            st.info("💡 Try clicking 'Setup' to configure API credentials first")
                    
                except Exception as e:
                    st.error(f"Auto-connect failed: {e}")
                
                del st.session_state["auto_connect_target"]
        
        # Handle pending fetch or setup
        pending = ctx.pending_fetch
        if pending:
            if pending.get("needs_setup"):
                st.warning(f"""
                **🔑 API Key Required**  
                
                To access this data source, you need an API key:
                1. Visit: [{pending.get('signup_url', 'API Website')}]({pending.get('signup_url', '#')})
                2. Create a free account and get your API key
                3. Set environment variable: `{pending.get('env_var', 'API_KEY')}`
                
                Or use the credential manager to store it securely.
                """)
                if st.button("✓ I've set up my API key"):
                    ctx.pending_fetch = None
                    st.rerun()
            else:
                st.info(f"Ready to fetch data from **{pending.get('api_id')}**. Use the Enrichment Options below to configure and fetch.")
                ctx.pending_fetch = None
    
    # Custom API section - for APIs not in the registry
    with st.expander("🔧 Custom API Connector", expanded=False):
        st.markdown("""
        **Don't see your API?** Generate a connector automatically from documentation.
        
        Provide a URL to API documentation (OpenAPI/Swagger spec or docs page) and we'll 
        create a connector for you.
        """)
        
        custom_docs_url = st.text_input(
            "API Documentation URL",
            placeholder="https://api.example.com/docs/swagger.json",
            key="custom_api_docs_url"
        )
        
        custom_api_key = st.text_input(
            "API Key (optional, for testing)",
            type="password",
            key="custom_api_key"
        )
        
        if st.button("🚀 Generate Connector", type="primary", disabled=not custom_docs_url):
            with st.spinner("Parsing documentation and generating connector..."):
                try:
                    from mcp_server.dynamic_connector import get_connector_manager
                    
                    manager = get_connector_manager()
                    result = manager.generate_connector(
                        custom_docs_url, 
                        api_key=custom_api_key if custom_api_key else None
                    )
                    
                    if result.validated:
                        st.success(f"✅ Generated connector for **{result.api_name}**")
                        st.markdown(f"""
                        **Base URL:** `{result.base_url}`  
                        **Auth Type:** {result.auth_type}  
                        **Endpoints Found:** {len(result.endpoints)}
                        """)
                        
                        with st.expander("View Generated Code"):
                            st.code(result.code, language="python")
                        
                        st.info("The connector is ready to use! You can now fetch data from this API.")
                    else:
                        st.error(f"❌ Generation failed: {result.error}")
                        st.info("💡 **Tip:** Try providing a direct link to an OpenAPI/Swagger specification (usually ends in .json or .yaml)")
                        
                except Exception as e:
                    st.error(f"Error: {e}")


def render_enrichment_options(df: pd.DataFrame):
    """
    Show buttons to trigger enrichment with real data sources.
    """
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("🌍 **World Bank**\n\nGDP, Population, Inflation")
        indicator = st.selectbox("Indicator", ["NY.GDP.MKTP.CD", "SP.POP.TOTL", "FP.CPI.TOTL.ZG"], key="wb_indicator",
                                  format_func=lambda x: {"NY.GDP.MKTP.CD": "GDP (USD)", "SP.POP.TOTL": "Population", "FP.CPI.TOTL.ZG": "Inflation %"}.get(x, x))
        if st.button("Connect World Bank"):
            with st.spinner("Fetching World Bank data..."):
                try:
                    from public_data.connectors.world_bank import WorldBankConnector
                    connector = WorldBankConnector()
                    # countries parameter is semicolon-separated string
                    wb_df = connector.fetch_data(indicator, countries="USA;GBR;CHN;JPN;DEU")
                    if wb_df is not None and not wb_df.empty:
                        st.session_state["enrichment_preview"] = wb_df
                        st.success(f"✅ Loaded {len(wb_df)} rows from World Bank")
                        st.rerun()
                    else:
                        st.warning("No data returned from World Bank")
                except Exception as e:
                    st.error(f"World Bank error: {e}")
    
    with col2:
        st.info("📈 **FRED Data**\n\nUnemployment, CPI, Fed Funds")
        fred_series = st.selectbox("Series", ["UNRATE", "CPIAUCSL", "FEDFUNDS", "GDP"], key="fred_series",
                                   format_func=lambda x: {"UNRATE": "Unemployment Rate", "CPIAUCSL": "CPI (Urban)", "FEDFUNDS": "Fed Funds Rate", "GDP": "US GDP"}.get(x, x))
        if st.button("Connect FRED"):
            with st.spinner("Fetching FRED data..."):
                try:
                    from public_data.connectors.fred import FREDConnector
                    connector = FREDConnector()
                    fred_df = connector.fetch_data(fred_series)
                    if fred_df is not None and not fred_df.empty:
                        st.session_state["enrichment_preview"] = fred_df
                        st.success(f"✅ Loaded {len(fred_df)} rows from FRED")
                        st.rerun()
                    else:
                        st.warning("No data returned from FRED")
                except Exception as e:
                    st.error(f"FRED error: {e}")
            
    with col3:
        st.info("🏥 **WHO Health**\n\n(Coming Soon)")
        st.button("Connect WHO", disabled=True)
    
    # Show enrichment preview if available
    if "enrichment_preview" in st.session_state:
        st.divider()
        st.subheader("📊 Enrichment Data Preview")
        st.dataframe(st.session_state["enrichment_preview"].head(20))
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("➕ Add to Datasets"):
                # Add as a new dataset
                dataset_name = f"Enrichment_{len(st.session_state.get('datasets', {}))}"
                if "datasets" not in st.session_state:
                    st.session_state["datasets"] = {}
                st.session_state["datasets"][dataset_name] = st.session_state["enrichment_preview"]
                st.success(f"Added as '{dataset_name}'")
                del st.session_state["enrichment_preview"]
                st.rerun()
        with col_b:
            if st.button("❌ Dismiss"):
                del st.session_state["enrichment_preview"]
                st.rerun()

