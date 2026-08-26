from __future__ import annotations
import streamlit as st
from pathlib import Path


def render_login_page() -> None:
    """
    Renders the full-screen login page.
    Called from app.py when st.user.is_logged_in is False.
    """

    # Load login-specific CSS safely
    try:
        _css = (Path(__file__).parent / "login.css").read_text(encoding="utf-8")
        st.markdown(f"<style>{_css}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

    # ── Logo ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="login-logo">🧮</div>', unsafe_allow_html=True)

    # ── Card ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="login-card">', unsafe_allow_html=True)

    st.markdown('<div class="login-title">数学智能辅导助手</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="login-subtitle">'
        'AI 分步解题，<br>'
        '根据你的学习情况个性化讲解。'
        '</div>',
        unsafe_allow_html=True,
    )

    # Feature pills
    st.markdown("""
    <div class="feature-row">
        <span class="feature-pill">📐 全知识点覆盖</span>
        <span class="feature-pill">🧠 跨会话记忆</span>
        <span class="feature-pill">🎬 可视化讲解</span>
        <span class="feature-pill">📄 上传学习资料</span>
    </div>
    """, unsafe_allow_html=True)

    # Google login button — st.login() handles the entire OIDC flow
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        if st.button("🔵  使用 Google 账号登录", use_container_width=True):
            st.login("google")

    # Divider
    st.markdown('<div class="login-divider">安全登录</div>', unsafe_allow_html=True)

    # Trust badges
    st.markdown("""
    <div class="trust-row">
        <span class="trust-item">🔒 Google OAuth 2.0 认证</span>
        <span class="trust-item">🛡️ 不存储密码</span>
        <span class="trust-item">☁️ 加密会话</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Footer
    st.markdown(
        '<div class="login-footer">登录即表示你同意仅将此工具用于学习目的。</div>',
        unsafe_allow_html=True,
    )