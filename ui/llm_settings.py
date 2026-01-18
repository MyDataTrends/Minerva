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
        with st.expander("🟢 OpenAI (GPT-4, GPT-4o)", expanded=True):
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
                if st.button("Use GPT-4o", disabled=not new_key):
                    if new_key:
                        os.environ["OPENAI_API_KEY"] = new_key
                    registry.set_active_model("gpt-4o")
                    st.success("Now using GPT-4o")
                    st.rerun()
            with col2:
                if st.button("Use GPT-4o Mini", disabled=not new_key):
                    if new_key:
                        os.environ["OPENAI_API_KEY"] = new_key
                    registry.set_active_model("gpt-4o-mini")
                    st.success("Now using GPT-4o Mini")
                    st.rerun()
        
        # Anthropic
        with st.expander("🟣 Anthropic (Claude 3.5)"):
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
                if st.button("Use Claude 3.5 Sonnet", disabled=not new_key):
                    if new_key:
                        os.environ["ANTHROPIC_API_KEY"] = new_key
                    registry.set_active_model("claude-3-5-sonnet")
                    st.success("Now using Claude 3.5 Sonnet")
                    st.rerun()
            with col2:
                if st.button("Use Claude 3 Haiku", disabled=not new_key):
                    if new_key:
                        os.environ["ANTHROPIC_API_KEY"] = new_key
                    registry.set_active_model("claude-3-haiku")
                    st.success("Now using Claude 3 Haiku")
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
