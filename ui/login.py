"""
登录界面模块
"""
import streamlit as st
from pathlib import Path


def load_user_credentials():
    """
    从 user.txt 文件加载用户凭证
    
    返回:
        dict: 用户信息字典 {"username": str, "password": str, "access_rights": str}
    """
    user_file = Path("user.txt")
    
    if not user_file.exists():
        st.error("用户配置文件不存在！")
        return None
    
    user_data = {}
    try:
        with open(user_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip().lower().replace(" ", "_")
                    value = value.strip().strip('"')
                    user_data[key] = value
        
        return user_data
    except Exception as e:
        st.error(f"读取用户配置失败: {str(e)}")
        return None


def verify_login(username, password):
    """
    验证用户登录
    
    参数:
        username (str): 用户名
        password (str): 密码
    
    返回:
        tuple: (bool, dict) - (是否验证成功, 用户信息)
    """
    user_data = load_user_credentials()
    
    if not user_data:
        return False, None
    
    if (user_data.get("username") == username and 
        user_data.get("password") == password):
        # 解析权限级别
        access_level = int(user_data.get("access_level", 1))
        
        return True, {
            "username": user_data.get("username"),
            "access_rights": user_data.get("access_rights", "user"),
            "access_level": access_level,  # ← 新增：权限级别
            "department": user_data.get("department", "")  # ← 新增：部门
        }
    
    return False, None


def render_login_page():
    """
    渲染登录页面
    """
    # 居中布局
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(
            """
            <div style='text-align: center; padding: 50px 0 30px 0;'>
                <h1>📊 AI BI Assistant V4</h1>
                <p style='color: #666; font-size: 16px;'>企业级智能商业分析助手</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # 登录表单
        with st.form("login_form"):
            st.markdown("### 🔐 用户登录")
            
            username = st.text_input(
                "用户名",
                placeholder="请输入用户名",
                key="login_username"
            )
            
            password = st.text_input(
                "密码",
                type="password",
                placeholder="请输入密码",
                key="login_password"
            )
            
            col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
            with col_btn2:
                submit_button = st.form_submit_button(
                    "登录",
                    use_container_width=True,
                    type="primary"
                )
            
            if submit_button:
                if not username or not password:
                    st.error("❌ 请输入用户名和密码！")
                else:
                    success, user_info = verify_login(username, password)
                    
                    if success:
                        # 保存登录状态到 session_state
                        st.session_state.logged_in = True
                        st.session_state.user_info = user_info
                        st.success("✅ 登录成功！正在跳转...")
                        st.rerun()
                    else:
                        st.error("❌ 用户名或密码错误！")
        
        # 底部提示
        st.markdown(
            """
            <div style='text-align: center; padding-top: 50px; color: #999; font-size: 12px;'>
                <p>💡 提示: 默认账号请查看 user.txt 文件</p>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_user_info_sidebar():
    """
    在侧边栏顶部渲染用户信息
    """
    if st.session_state.get("logged_in", False):
        user_info = st.session_state.get("user_info", {})
        username = user_info.get("username", "未知用户")
        access_rights = user_info.get("access_rights", "user")
        access_level = user_info.get("access_level", 1)
        department = user_info.get("department", "")
        
        # 权限级别映射
        level_names = {
            1: "一级权限 (基础)",
            2: "二级权限 (中级)",
            3: "三级权限 (高级)"
        }
        level_name = level_names.get(access_level, "未知级别")
        
        # 权限等级颜色映射
        level_colors = {
            1: "#83c9ff",  # 蓝色 - 基础
            2: "#0068c9",  # 深蓝 - 中级
            3: "#ff4b4b"   # 红色 - 高级
        }
        level_color = level_colors.get(access_level, "#666")
        
        st.sidebar.markdown(
            f"""
            <div style='
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            '>
                <div style='color: white; text-align: center;'>
                    <div style='font-size: 24px; margin-bottom: 5px;'>👤</div>
                    <div style='font-size: 16px; font-weight: bold; margin-bottom: 5px;'>{username}</div>
                    <div style='
                        display: inline-block;
                        background: rgba(255,255,255,0.3);
                        padding: 3px 12px;
                        border-radius: 12px;
                        font-size: 12px;
                        font-weight: 500;
                        margin-bottom: 5px;
                    '>
                        {access_rights.upper()}
                    </div>
                    <div style='
                        font-size: 11px;
                        opacity: 0.9;
                        margin-top: 5px;
                    '>
                        🔐 {level_name}
                    </div>
                    {f"<div style='font-size: 10px; opacity: 0.8; margin-top: 3px;'>📁 {department}</div>" if department else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # 登出按钮
        if st.sidebar.button("🚪 退出登录", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_info = {}
            st.rerun()


def check_login_status():
    """
    检查登录状态
    
    返回:
        bool: 是否已登录
    """
    return st.session_state.get("logged_in", False)
