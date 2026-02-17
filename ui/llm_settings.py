"""
LLM Settings UI - Streamlit component for managing LLM models.

Provides a settings interface for:
- Scanning for local models
- Selecting active model
- Configuring API keys
- Downloading open source models
"""
import streamlit as st
import os
from pathlib import Path
from typing import Optional

# Lazy imports to avoid circular dependencies
def _load_env():
    try:
        from utils.env_loader import load_env
        load_env()
    except ImportError:
        pass


def _get_registry():
    from llm_manager.registry import get_registry
    return get_registry()


def _get_provider_base():
    from llm_manager.providers.base import ProviderType, DOWNLOADABLE_MODELS
    return ProviderType, DOWNLOADABLE_MODELS


def render_llm_settings():
    """
    Render the LLM settings panel.
    
    Can be used as a standalone page or embedded in another page.
    """
    _load_env()
    registry = _get_registry()
    ProviderType, DOWNLOADABLE_MODELS = _get_provider_base()
    
    st.header("🤖 LLM Model Manager")
    st.caption("Configure which AI model powers Minerva's analysis")
    
    # Tabs for different sections
    local_tab, cloud_tab, download_tab = st.tabs([
        "📁 Local Models",
        "☁️ Cloud APIs", 
        "⬇️ Download Models"
    ])
    
    # === LOCAL MODELS TAB ===
    with local_tab:
        st.subheader("Local Models")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.write("Scan your system for installed LLM models (GGUF format)")
        with col2:
            if st.button("🔍 Scan for Models", type="primary"):
                with st.spinner("Scanning..."):
                    count = registry.scan_local_models()
                    if count > 0:
                        st.success(f"Found {count} local model(s)!")
                        st.rerun()
                    else:
                        st.info("No local models found. Download one or add a custom path.")
        
        # Custom path input
        with st.expander("Add custom model path"):
            custom_path = st.text_input(
                "Model file or directory",
                placeholder="C:/models/my-model.gguf"
            )
            if custom_path and st.button("Add Path"):
                path = Path(custom_path)
                if path.exists():
                    if path.is_file() and path.suffix == ".gguf":
                        from llm_manager.providers.base import ModelInfo, ProviderType as PT
                        model_info = ModelInfo(
                            id=f"local:{path.stem}",
                            name=path.stem.replace("-", " ").title(),
                            provider_type=PT.LOCAL,
                            path_or_endpoint=str(path),
                            description="Custom local model",
                            size_gb=round(path.stat().st_size / (1024**3), 1),
                        )
                        registry.register_model(model_info)
                        st.success(f"Added model: {model_info.name}")
                        st.rerun()
                    elif path.is_dir():
                        count = registry.scan_local_models([path])
                        st.success(f"Scanned directory, found {count} model(s)")
                        st.rerun()
                else:
                    st.error("Path not found")
        
        st.divider()
        
        # List local models
        local_models = registry.get_local_models()
        ollama_models = registry.get_ollama_models()
        
        if not local_models and not ollama_models:
            st.info("No local models found. Click 'Scan for Models' or download one below.")
        else:
            st.write(f"**{len(local_models)} local model(s) available:**")
            
            active = registry.get_active_model()
            
            for model in local_models + ollama_models:
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    is_active = active and active.id == model.id
                    status = "✅ Active" if is_active else ""
                    size = f"{model.size_gb:.1f} GB" if model.size_gb else ""
                    st.write(f"**{model.name}** {status}")
                    st.caption(f"{model.description} • {size}")
                
                with col2:
                    if not (active and active.id == model.id):
                        if st.button("Use", key=f"use_{model.id}"):
                            registry.set_active_model(model.id)
                            st.success(f"Now using: {model.name}")
                            st.rerun()
                
                with col3:
                    with st.popover("ℹ️"):
                        st.write(f"**Path:** `{model.path_or_endpoint}`")
                        st.write(f"**Context:** {model.context_length} tokens")
    
    # === CLOUD APIS TAB ===
    with cloud_tab:
        st.subheader("Cloud API Providers")
        st.write("Use cloud-hosted models via API")
        
        # OpenAI
        with st.expander("🟢 OpenAI (GPT-5.3)", expanded=True):
            openai_key = os.getenv("OPENAI_API_KEY", "")
            new_key = st.text_input(
                "OpenAI API Key",
                value=openai_key if openai_key else "",
                type="password",
                key="openai_key"
            )
            
            if new_key and new_key != openai_key:
                st.info("Save this key to your .env file: `OPENAI_API_KEY=your_key`")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Use GPT-5.3", disabled=not new_key):
                    if new_key:
                        os.environ["OPENAI_API_KEY"] = new_key
                    registry.set_active_model("gpt-5.3")
                    st.success("Now using GPT-5.3")
                    st.rerun()
            with col2:
                if st.button("Use GPT-5.3 Codex", disabled=not new_key):
                    if new_key:
                        os.environ["OPENAI_API_KEY"] = new_key
                    registry.set_active_model("gpt-5.3-codex")
                    st.success("Now using GPT-5.3 Codex")
                    st.rerun()
        
        # Anthropic
        with st.expander("🟣 Anthropic (Claude 4.5/4.6)"):
            anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
            new_key = st.text_input(
                "Anthropic API Key",
                value=anthropic_key if anthropic_key else "",
                type="password",
                key="anthropic_key"
            )
            
            if new_key and new_key != anthropic_key:
                st.info("Save this key to your .env file: `ANTHROPIC_API_KEY=your_key`")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Use Claude 4.5 Sonnet", disabled=not new_key):
                    if new_key:
                        os.environ["ANTHROPIC_API_KEY"] = new_key
                    registry.set_active_model("claude-4-5-sonnet")
                    st.success("Now using Claude 4.5 Sonnet")
                    st.rerun()
            with col2:
                if st.button("Use Claude 4.5 Haiku", disabled=not new_key):
                    if new_key:
                        os.environ["ANTHROPIC_API_KEY"] = new_key
                    registry.set_active_model("claude-4-5-haiku")
                    st.success("Now using Claude 4.5 Haiku")
                    st.rerun()
        
        # Ollama
        with st.expander("🦙 Ollama (Local Server)"):
            st.write("Use Ollama for running models locally with easy management")
            st.caption("Install from: https://ollama.ai")
            
            if st.button("Check Ollama"):
                from llm_manager.scanner import check_ollama_models
                models = check_ollama_models()
                if models:
                    st.success(f"Found {len(models)} Ollama model(s)")
                    for m in models:
                        st.write(f"- {m.name}")
                else:
                    st.warning("Ollama not running or no models installed")
        
        # Kaggle (for API discovery weighting)
        with st.expander("📊 Kaggle (Data Discovery)", expanded=False):
            st.write("Kaggle credentials enable smart API discovery based on popular datasets")
            
            # Help section - updated for current Kaggle UI
            with st.popover("ℹ️ How to get your Kaggle API key"):
                st.markdown("""
                ### Step-by-Step Setup
                
                1. **Go to** [kaggle.com/settings](https://www.kaggle.com/settings)
                2. **Scroll to** the "API" section
                3. **Click** "Create New Token" (or view existing)
                4. **Copy** your username and API key from the page
                5. **Paste** them into the fields below
                6. **Create** a master password to encrypt your credentials
                
                ---
                
                **What's the master password?**  
                A password you create to encrypt your credentials on this computer.
                You'll enter it once per session to unlock access to Kaggle features.
                """)
            
            # Check current status
            try:
                from mcp_server.credential_manager import (
                    get_kaggle_credential_status,
                    store_kaggle_credentials,
                    get_kaggle_credentials,
                    has_kaggle_credentials
                )
                
                status = get_kaggle_credential_status()
                
                # Check if already unlocked this session
                session_unlocked = st.session_state.get("kaggle_unlocked", False)
                
                if status.get("available"):
                    source = status.get("source", "unknown")
                    username = status.get("username", "unknown")
                    
                    source_labels = {
                        "environment": "Environment variables",
                        "kaggle_json": "~/.kaggle/kaggle.json",
                        "encrypted_storage": "Minerva encrypted storage"
                    }
                    
                    # If from encrypted storage, may need unlock
                    if source == "encrypted_storage":
                        # Verify the session password actually works before showing "unlocked"
                        session_password = st.session_state.get("kaggle_master_password")
                        
                        # Check if we have a verified working password
                        password_verified = False
                        if session_password:
                            try:
                                test_creds = get_kaggle_credentials(session_password)
                                if test_creds and test_creds.get("username"):
                                    password_verified = True
                            except:
                                pass
                        
                        if password_verified:
                            st.success(f"🔓 Unlocked for this session")
                            st.caption(f"Username: {username}")
                            
                            # Option to clear credentials
                            if st.button("🗑️ Clear stored credentials", key="clear_kaggle"):
                                from mcp_server.credential_manager import CredentialManager, KAGGLE_API_ID
                                cred_mgr = CredentialManager()
                                cred_mgr.delete_credential(KAGGLE_API_ID)
                                st.session_state.pop("kaggle_unlocked", None)
                                st.session_state.pop("kaggle_master_password", None)
                                st.success("Cleared! Re-enter credentials below.")
                                st.rerun()
                        else:
                            st.warning(f"🔒 Credentials stored (encrypted)")
                            st.caption(f"Username: {username}")
                            
                            # Unlock form
                            unlock_pw = st.text_input(
                                "Enter Master Password to unlock",
                                type="password",
                                key="kaggle_unlock_pw"
                            )
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🔓 Unlock", disabled=not unlock_pw):
                                    test_creds = get_kaggle_credentials(unlock_pw)
                                    if test_creds and test_creds.get("username"):
                                        st.session_state["kaggle_unlocked"] = True
                                        st.session_state["kaggle_master_password"] = unlock_pw
                                        st.success("✅ Credentials unlocked!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Wrong password - decryption failed")
                            
                            with col2:
                                if st.button("🗑️ Clear & re-enter"):
                                    from mcp_server.credential_manager import CredentialManager, KAGGLE_API_ID
                                    cred_mgr = CredentialManager()
                                    cred_mgr.delete_credential(KAGGLE_API_ID)
                                    st.session_state.pop("kaggle_unlocked", None)
                                    st.session_state.pop("kaggle_master_password", None)
                                    st.rerun()
                    else:
                        # Env vars or kaggle.json - no unlock needed
                        st.success(f"✅ Configured ({source_labels.get(source, source)})")
                        st.caption(f"Username: {username}")
                        st.session_state["kaggle_unlocked"] = True
                    
                else:
                    st.info("Kaggle credentials not configured")
                    
                    # Input form for new credentials
                    kaggle_user = st.text_input(
                        "Kaggle Username",
                        key="kaggle_username_input",
                        help="Your Kaggle username"
                    )
                    kaggle_key = st.text_input(
                        "Kaggle API Key", 
                        type="password",
                        key="kaggle_key_input",
                        help="Your Kaggle API key (from Settings → API)"
                    )
                    master_pw = st.text_input(
                        "Create Master Password",
                        type="password",
                        key="kaggle_master_pw",
                        help="This password encrypts your credentials on your computer"
                    )
                    
                    if st.button("💾 Save Credentials", disabled=not (kaggle_user and kaggle_key and master_pw)):
                        try:
                            store_kaggle_credentials(
                                username=kaggle_user,
                                api_key=kaggle_key,
                                master_password=master_pw
                            )
                            st.session_state["kaggle_unlocked"] = True
                            st.session_state["kaggle_master_password"] = master_pw
                            st.success(f"✅ Saved and unlocked for {kaggle_user}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to save: {e}")
                            
            except ImportError as e:
                st.warning(f"Credential manager not available: {e}")

        # ===== CUSTOM API KEYS SECTION =====
        with st.expander("🔑 Custom API Keys (Data Sources)", expanded=False):
            st.write("Store API keys for data services discovered by Minerva.")
            
            with st.popover("ℹ️ How this works"):
                st.markdown("""
                ### Secure API Key Storage
                
                When Minerva discovers an API that requires authentication 
                (like ESPN, Football-Data.org, or Alpha Vantage), you can 
                store your API key here.
                
                **Your keys are:**
                - 🔐 Encrypted with your master password
                - 📁 Stored locally in `~/.minerva/credentials.json`
                - 🔓 Unlocked per session (same as Kaggle)
                
                **To get an API key:**
                1. Visit the API's website and create an account
                2. Look for "API Keys" or "Developer" section
                3. Generate or copy your API key
                4. Paste it below with a descriptive API ID
                """)
            
            try:
                from mcp_server.credential_manager import CredentialManager
                
                cred_mgr = CredentialManager()
                stored_creds = cred_mgr.list_credentials()
                
                # Filter out Kaggle (handled separately)
                custom_creds = [c for c in stored_creds if c.get("api_id") != "kaggle"]
                
                # Show stored keys
                if custom_creds:
                    st.write(f"**Stored API Keys:** {len(custom_creds)}")
                    
                    for cred in custom_creds:
                        api_id = cred.get("api_id", "unknown")
                        created = cred.get("created_at", "")[:10]  # Just date
                        metadata = cred.get("metadata", {})
                        
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.write(f"🔐 **{api_id}**")
                            if metadata:
                                st.caption(f"Added: {created}")
                        with col2:
                            if st.button("🗑️", key=f"del_{api_id}", help=f"Delete {api_id}"):
                                cred_mgr.delete_credential(api_id)
                                st.success(f"Deleted {api_id}")
                                st.rerun()
                    
                    st.divider()
                
                # Add new API key
                st.write("**Add New API Key**")
                
                # Common APIs for suggestions
                common_apis = {
                    "espn": "ESPN API",
                    "football_data": "Football-Data.org", 
                    "alpha_vantage": "Alpha Vantage (Stocks)",
                    "openweathermap": "OpenWeatherMap",
                    "newsapi": "NewsAPI",
                    "tmdb": "TMDb (Movies)",
                    "custom": "Custom (enter below)"
                }
                
                selected_api = st.selectbox(
                    "Select API",
                    options=list(common_apis.keys()),
                    format_func=lambda x: common_apis[x],
                    key="settings_api_select",
                    help="Choose a common API or select 'Custom'"
                )
                
                if selected_api == "custom":
                    api_id = st.text_input(
                        "Custom API ID", 
                        key="settings_custom_api_id",
                        help="A short identifier like 'my_service'"
                    )
                else:
                    api_id = selected_api
                    
                api_key = st.text_input(
                    "API Key",
                    type="password",
                    key="settings_custom_api_key",
                    help="Your API key from the service"
                )
                
                master_pw = st.text_input(
                    "Master Password",
                    type="password", 
                    key="settings_custom_master_pw",
                    help="Same password you use for Kaggle credentials"
                )
                
                if st.button("💾 Save API Key", disabled=not (api_id and api_key and master_pw)):
                    try:
                        cred_mgr.store_credential(
                            api_id=api_id,
                            api_key=api_key,
                            master_password=master_pw
                        )
                        st.success(f"✅ Saved {api_id} API key (encrypted)")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to save: {e}")
                        
            except ImportError as e:
                st.warning(f"Credential manager not available: {e}")
    
    # === DOWNLOAD TAB ===
    with download_tab:
        st.subheader("Download Open Source Models")
        st.write("Download popular open source models for local use")
        
        st.warning("⚠️ Models are large files (2-5 GB). Ensure you have sufficient disk space.")
        
        for model_id, info in DOWNLOADABLE_MODELS.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.write(f"**{info['name']}**")
                st.caption(f"{info['description']} • {info['size_gb']} GB")
            
            with col2:
                st.write(f"{info['size_gb']} GB")
            
            with col3:
                if st.button("Download", key=f"dl_{model_id}"):
                    from llm_manager.downloader import download_model, create_model_info_for_download
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(percent, status):
                        progress_bar.progress(int(percent))
                        status_text.text(status)
                    
                    path = download_model(model_id, progress_callback=update_progress)
                    
                    if path:
                        model_info = create_model_info_for_download(model_id, path)
                        registry.register_model(model_info)
                        st.success(f"Downloaded: {info['name']}")
                        st.rerun()
                    else:
                        st.error("Download failed")
    
    # === CURRENT STATUS ===
    st.divider()
    active = registry.get_active_model()
    if active:
        st.success(f"**Active Model:** {active.name}")
        st.caption(f"Type: {active.provider_type.value} | Context: {active.context_length} tokens")
    else:
        st.warning("No model selected. Choose a model above to enable AI features.")


def render_llm_settings_compact():
    """
    Render a compact version of LLM settings for embedding in other panels.
    """
    _load_env()
    registry = _get_registry()
    
    active = registry.get_active_model()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        if active:
            st.write(f"🤖 **{active.name}**")
        else:
            st.write("🤖 No model selected")
    
    with col2:
        if st.button("⚙️ Settings"):
            st.session_state["show_llm_settings"] = True


# For testing standalone
if __name__ == "__main__":
    st.set_page_config(page_title="LLM Settings", layout="wide")
    render_llm_settings()
