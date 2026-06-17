"""
Authentication and User Management
Malaria Forecast System — SW Cameroon
======================================
Handles: login, logout, add user, manage users
Uses: SQLite + hashlib (no extra installs needed)
"""

import streamlit as st
import sqlite3
import hashlib
import os
from datetime import datetime


# ─────────────────────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────────────────────
def init_users_db():
    """Create users table if it does not exist.
    Also creates the default admin account on first run."""
    conn = sqlite3.connect("malaria_forecasts.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT    NOT NULL UNIQUE,
            full_name   TEXT    NOT NULL,
            role        TEXT    NOT NULL DEFAULT 'health_officer',
            district    TEXT,
            password_hash TEXT  NOT NULL,
            created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active   INTEGER DEFAULT 1
        )
    """)
    conn.commit()

    # Create default admin if no users exist
    count = conn.execute(
        "SELECT COUNT(*) FROM users").fetchone()[0]
    if count == 0:
        admin_hash = hash_password("admin123")
        conn.execute("""
            INSERT INTO users
            (username, full_name, role, district, password_hash)
            VALUES (?, ?, ?, ?, ?)
        """, ("admin", "System Administrator",
              "admin", "All Districts", admin_hash))
        conn.commit()
    conn.close()


def hash_password(password: str) -> str:
    """Hash a password using SHA-256."""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_login(username: str, password: str):
    """Check username and password. Returns user dict or None."""
    conn = sqlite3.connect("malaria_forecasts.db")
    row = conn.execute("""
        SELECT id, username, full_name, role, district
        FROM users
        WHERE username = ?
          AND password_hash = ?
          AND is_active = 1
    """, (username.strip().lower(),
          hash_password(password))).fetchone()
    conn.close()
    if row:
        return {
            "id":        row[0],
            "username":  row[1],
            "full_name": row[2],
            "role":      row[3],
            "district":  row[4],
        }
    return None


def add_user(username, full_name, role,
             district, password, created_by):
    """Add a new user. Returns (success, message)."""
    if not username or not full_name or not password:
        return False, "Username, full name, and password are required."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    try:
        conn = sqlite3.connect("malaria_forecasts.db")
        conn.execute("""
            INSERT INTO users
            (username, full_name, role, district, password_hash)
            VALUES (?, ?, ?, ?, ?)
        """, (username.strip().lower(),
              full_name.strip(), role,
              district, hash_password(password)))
        conn.commit()
        conn.close()
        return True, f"User '{username}' created successfully."
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' already exists."
    except Exception as e:
        return False, f"Error: {str(e)}"


def get_all_users():
    """Return all users as a list of dicts."""
    conn = sqlite3.connect("malaria_forecasts.db")
    rows = conn.execute("""
        SELECT id, username, full_name, role,
               district, is_active, created_at
        FROM users
        ORDER BY created_at DESC
    """).fetchall()
    conn.close()
    return [
        {"id": r[0], "username": r[1], "full_name": r[2],
         "role": r[3], "district": r[4],
         "active": bool(r[5]), "created_at": r[6]}
        for r in rows
    ]


def deactivate_user(user_id: int):
    """Deactivate a user (soft delete)."""
    conn = sqlite3.connect("malaria_forecasts.db")
    conn.execute(
        "UPDATE users SET is_active = 0 WHERE id = ?",
        (user_id,))
    conn.commit()
    conn.close()


def reactivate_user(user_id: int):
    """Reactivate a deactivated user."""
    conn = sqlite3.connect("malaria_forecasts.db")
    conn.execute(
        "UPDATE users SET is_active = 1 WHERE id = ?",
        (user_id,))
    conn.commit()
    conn.close()


def change_password(user_id: int,
                    old_password: str,
                    new_password: str):
    """Change a user's password. Returns (success, message)."""
    if len(new_password) < 6:
        return False, "New password must be at least 6 characters."
    conn = sqlite3.connect("malaria_forecasts.db")
    row = conn.execute(
        "SELECT id FROM users WHERE id=? AND password_hash=?",
        (user_id, hash_password(old_password))).fetchone()
    if not row:
        conn.close()
        return False, "Current password is incorrect."
    conn.execute(
        "UPDATE users SET password_hash=? WHERE id=?",
        (hash_password(new_password), user_id))
    conn.commit()
    conn.close()
    return True, "Password changed successfully."


# ─────────────────────────────────────────────────────────────
# STREAMLIT PAGES
# ─────────────────────────────────────────────────────────────
def show_login_page():
    """Render the login page. Sets st.session_state.user on success."""

    # CSS for login page
    st.markdown("""
    <style>
    .login-header {
        text-align:center; padding:2rem 0 1rem;
        color:#2C5F2D; font-size:2rem; font-weight:700;
    }
    .login-sub {
        text-align:center; color:#6B7B6B;
        font-size:1rem; margin-bottom:2rem;
    }
    .login-box {
        max-width:420px; margin:0 auto;
        background:#F5F9F5; padding:2rem 2.5rem;
        border-radius:16px; border:1px solid #D8E8D8;
        box-shadow:0 4px 20px rgba(44,95,45,0.08);
    }
    </style>
    """, unsafe_allow_html=True)

    # Centre the form
    _, centre, _ = st.columns([1, 2, 1])
    with centre:
        st.markdown(
            '<div class="login-header">🦟 Malaria Forecast</div>'
            '<div class="login-sub">Southwest Cameroon Health Districts'
            '<br>Decision Support System</div>',
            unsafe_allow_html=True)

        with st.form("login_form"):
            st.markdown("#### Sign In")
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                autocomplete="username")
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                autocomplete="current-password")
            submit = st.form_submit_button(
                "🔐 Sign In", use_container_width=True)

        if submit:
            if not username or not password:
                st.error("Please enter both username and password.")
            else:
                user = verify_login(username, password)
                if user:
                    st.session_state["user"] = user
                    st.session_state["logged_in"] = True
                    st.success(
                        f"Welcome, {user['full_name']}!")
                    st.rerun()
                else:
                    st.error(
                        "Incorrect username or password. "
                        "Contact your administrator if you have "
                        "forgotten your credentials.")

        st.markdown("---")
        st.caption(
            "Default admin login: **admin** / **admin123**  \n"
            "Please change the default password after first login.")

        st.markdown(
            "<div style='text-align:center;color:#6B7B6B;"
            "font-size:0.8rem;margin-top:1rem'>"
            "University of Buea · Dept of Computer Engineering<br>"
            "NDOUKIE EBOKE BLANDINE · FE22A254</div>",
            unsafe_allow_html=True)


def show_add_user_page(current_user: dict):
    """Page for adding new users. Admin only."""
    DISTRICTS = ["All Districts","Buea","Limbe",
                 "Muyuka","Tiko","Kumba"]
    ROLES = {
        "health_officer": "Health Officer",
        "district_manager": "District Manager",
        "admin": "System Administrator"
    }

    st.markdown("""
    <style>
    .section-header {
        font-size:1.3rem; font-weight:600; color:#2C5F2D;
        border-left:4px solid #2C5F2D; padding-left:0.7rem;
        margin:1.5rem 0 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        "## 👥 User Management",
        unsafe_allow_html=False)
    st.caption(
        "Manage who can access the Malaria Forecast System.")

    if current_user["role"] != "admin":
        st.warning(
            "Only system administrators can manage users.")
        return

    tab_add, tab_list, tab_pwd = st.tabs(
        ["➕ Add New User",
         "📋 All Users",
         "🔑 Change Password"])

    # ── TAB 1: Add new user ────────────────────────────────────
    with tab_add:
        st.markdown(
            '<div class="section-header">Add New User</div>',
            unsafe_allow_html=True)

        col_form, col_guide = st.columns([3, 2])
        with col_form:
            with st.form("add_user_form"):
                full_name = st.text_input(
                    "Full Name *",
                    placeholder="e.g. Dr. Jean Pierre Nkeng")
                username = st.text_input(
                    "Username *",
                    placeholder="e.g. jpnkeng  (lowercase, no spaces)")
                role = st.selectbox(
                    "Role *",
                    options=list(ROLES.keys()),
                    format_func=lambda x: ROLES[x])
                district = st.selectbox(
                    "Assigned District",
                    DISTRICTS,
                    help="Which district this user manages")
                password = st.text_input(
                    "Password *",
                    type="password",
                    placeholder="Minimum 6 characters",
                    help="The user can change this after first login")
                confirm = st.text_input(
                    "Confirm Password *",
                    type="password",
                    placeholder="Re-enter password")
                submitted = st.form_submit_button(
                    "✅ Create User",
                    use_container_width=True)

            if submitted:
                if password != confirm:
                    st.error(
                        "Passwords do not match. Please re-enter.")
                else:
                    ok, msg = add_user(
                        username, full_name, role,
                        district, password,
                        current_user["username"])
                    if ok:
                        st.success(msg)
                        st.balloons()
                    else:
                        st.error(msg)

        with col_guide:
            st.markdown("""
**Role permissions:**

🔵 **Health Officer**
- Can generate forecasts
- Can view trends and history
- Cannot manage users

🟡 **District Manager**
- All health officer permissions
- Can view all districts
- Cannot manage users

🔴 **System Administrator**
- Full access to all features
- Can add and manage users
- Can clear forecast history
            """)
            st.info(
                "After creating an account, share the username "
                "and password with the new user and ask them to "
                "change their password on first login.")

    # ── TAB 2: List all users ──────────────────────────────────
    with tab_list:
        st.markdown(
            '<div class="section-header">All Registered Users</div>',
            unsafe_allow_html=True)

        users = get_all_users()
        if not users:
            st.info("No users found.")
        else:
            for u in users:
                active_icon = "🟢" if u["active"] else "🔴"
                role_badge = {
                    "admin": "🔴 Admin",
                    "district_manager": "🟡 District Manager",
                    "health_officer": "🔵 Health Officer"
                }.get(u["role"], u["role"])

                with st.container():
                    c1, c2, c3, c4, c5 = st.columns(
                        [3, 2, 2, 2, 1])
                    with c1:
                        st.markdown(
                            f"**{u['full_name']}**  \n"
                            f"`@{u['username']}`")
                    with c2:
                        st.markdown(role_badge)
                    with c3:
                        st.markdown(
                            f"📍 {u['district'] or 'N/A'}")
                    with c4:
                        st.markdown(
                            f"{active_icon} "
                            f"{'Active' if u['active'] else 'Inactive'}")
                    with c5:
                        # Don't let admin deactivate themselves
                        if u["username"] != current_user["username"]:
                            if u["active"]:
                                if st.button(
                                        "Deactivate",
                                        key=f"deact_{u['id']}"):
                                    deactivate_user(u["id"])
                                    st.rerun()
                            else:
                                if st.button(
                                        "Reactivate",
                                        key=f"react_{u['id']}"):
                                    reactivate_user(u["id"])
                                    st.rerun()
                    st.divider()

    # ── TAB 3: Change password ─────────────────────────────────
    with tab_pwd:
        st.markdown(
            '<div class="section-header">'
            'Change Your Password</div>',
            unsafe_allow_html=True)

        with st.form("change_pwd_form"):
            old_pwd = st.text_input(
                "Current Password", type="password")
            new_pwd = st.text_input(
                "New Password",
                type="password",
                help="Minimum 6 characters")
            confirm_pwd = st.text_input(
                "Confirm New Password", type="password")
            change = st.form_submit_button(
                "🔑 Change Password",
                use_container_width=True)

        if change:
            if new_pwd != confirm_pwd:
                st.error("New passwords do not match.")
            else:
                ok, msg = change_password(
                    current_user["id"], old_pwd, new_pwd)
                if ok:
                    st.success(msg)
                else:
                    st.error(msg)
