# 分级权限知识库系统 - 快速使用指南

## 🚀 快速开始

### 1. 重建 Milvus Collection（必须！）
由于 Schema 增加了新字段，需要重建集合：

```bash
# 方法 1: 使用清理脚本
python tests/clear_milvus.py

# 方法 2: 手动删除（如果方法1失败）
# 删除 volumes/milvus/ 目录下所有内容
# 重启 Milvus
docker-compose -f docker-compose.milvus.yml restart
```

### 2. 启动系统
```bash
# 启动 Milvus
docker-compose -f docker-compose.milvus.yml up -d

# 启动应用
streamlit run main.py
```

### 3. 登录
- **用户名**: `Hzvy`
- **密码**: `123456`
- **权限级别**: 3 (高级管理员)

## 📤 上传文档

### 步骤
1. 登录后，左侧边栏 → **📚 知识库** → **📤 上传文件**
2. 选择知识库级别：
   - **🌐 一级 - 公开**: 所有用户可见（产品手册、FAQ）
   - **🔒 二级 - 内部**: 二级及以上权限（内部流程、技术文档）
   - **🔐 三级 - 机密**: 仅三级管理员（战略规划、财务数据）
3. 上传文件（支持 .txt, .md, .pdf）
4. 点击 **🚀 上传并更新知识库**

### 权限验证
- ✅ Level 3 用户可以上传任意级别文档
- ❌ Level 2 用户尝试上传 Level 3 文档 → 系统阻止
- ❌ Level 1 用户尝试上传 Level 2/3 文档 → 系统阻止

## 🔍 测试权限过滤

### 准备测试文档
创建三个文本文件：

**public.txt** (Level 1 - 公开)
```
这是公开文档。
所有用户都可以查看产品功能和使用手册。
```

**internal.txt** (Level 2 - 内部)
```
这是内部文档。
包含公司内部流程和技术实现细节。
仅限内部员工访问。
```

**confidential.txt** (Level 3 - 机密)
```
这是机密文档。
包含公司战略规划和核心技术秘密。
仅限高级管理人员访问。
```

### 上传测试
1. 选择 **Level 1**，上传 `public.txt`
2. 选择 **Level 2**，上传 `internal.txt`
3. 选择 **Level 3**，上传 `confidential.txt`

### 查询测试

#### 测试 1: Level 3 用户（查看所有）
**查询**: "请检索所有文档"

**预期结果**: 返回 3 个文档
- ✅ public.txt (Level 1)
- ✅ internal.txt (Level 2)
- ✅ confidential.txt (Level 3)

#### 测试 2: 创建 Level 1 用户
修改 `user.txt`:
```
username = "guest"
password = "123456"
access_rights = "user"
access_level = 1
department = "访客"
```

退出登录 → 重新登录 → 查询 "请检索所有文档"

**预期结果**: 仅返回 1 个文档
- ✅ public.txt (Level 1)
- ❌ internal.txt (Level 2) - 权限不足
- ❌ confidential.txt (Level 3) - 权限不足

## 🧪 自动化测试

运行测试脚本验证权限系统：
```bash
python tests/test_permission_system.py
```

**测试输出示例**:
```
============================================================
测试分级权限知识库系统
============================================================

1. 连接 Milvus...
🔌 连接 Milvus: localhost:19530

2. 检查集合...
✅ 集合已就绪

3. 插入测试文档...
✅ 插入 3 个文档

============================================================
4. 测试权限过滤
============================================================

【Level 1 用户检索】
🔍 混合检索: '文档' (Top 10, Level 1)
🔒 权限过滤: kb_level <= 1
检索到 1 个结果：
  - Level 1: 这是公开文档，所有用户都可以看到...

【Level 2 用户检索】
🔍 混合检索: '文档' (Top 10, Level 2)
🔒 权限过滤: kb_level <= 2
检索到 2 个结果：
  - Level 1: 这是公开文档，所有用户都可以看到...
  - Level 2: 这是内部文档，仅二级及以上权限可见...

【Level 3 用户检索】
🔍 混合检索: '文档' (Top 10, Level 3)
🔒 权限过滤: kb_level <= 3
检索到 3 个结果：
  - Level 1: 这是公开文档，所有用户都可以看到...
  - Level 2: 这是内部文档，仅二级及以上权限可见...
  - Level 3: 这是机密文档，仅三级管理员可见...

============================================================
5. 验证结果
============================================================
✅ Level 1 权限测试通过：仅返回公开文档
✅ Level 2 权限测试通过：返回公开+内部文档
✅ Level 3 权限测试通过：返回所有文档

============================================================
🎉 所有测试通过！权限系统工作正常
============================================================
```

## 👥 多用户管理

### 创建不同权限的用户

#### 方法 1: 修改 user.txt（简单）
```ini
# Level 3 管理员
username = "admin"
password = "admin123"
access_rights = "admin"
access_level = 3
department = "管理层"

# Level 2 普通用户
username = "user"
password = "user123"
access_rights = "user"
access_level = 2
department = "技术部"

# Level 1 访客
username = "guest"
password = "guest123"
access_rights = "guest"
access_level = 1
department = "访客"
```

**注意**: 当前系统仅支持单用户，需要手动切换。

#### 方法 2: 扩展为多用户系统（待开发）
创建 `users.json`:
```json
{
  "users": [
    {
      "username": "admin",
      "password_hash": "...",
      "access_level": 3,
      "department": "管理层"
    },
    {
      "username": "user",
      "password_hash": "...",
      "access_level": 2,
      "department": "技术部"
    }
  ]
}
```

## 📝 日志查看

### 检索日志
查看用户检索时的权限过滤日志：
```bash
# Windows
type logs\app.log | findstr "权限过滤"

# Linux/Mac
grep "权限过滤" logs/app.log
```

**示例输出**:
```
2025-10-11 10:30:15 - 🔍 混合检索: '技术文档' (Top 5, Level 2)
2025-10-11 10:30:15 - 🔒 权限过滤: kb_level <= 2
2025-10-11 10:30:16 - ✅ 检索到 3 个结果（权限级别 ≤ 2）
```

## ❓ 常见问题

### Q1: 上传后查询不到文档？
**A**: 检查：
1. 点击了 "🔄 重新加载知识库" 吗？
2. 查询权限是否足够？（Level 1 用户看不到 Level 2/3 文档）
3. Milvus 是否正常运行？ `docker ps`

### Q2: 权限过滤不生效？
**A**: 确认：
1. 已重建 Milvus Collection（新 Schema）
2. 文档是用新系统上传的（旧文档默认 Level 1）
3. 查看日志确认 `kb_level` 字段存在

### Q3: 如何查看文档的权限级别？
**A**: 暂时无 UI 界面，可以：
```python
# Python 脚本查询
from src.vector_db.milvus_manager import MilvusManager
milvus = MilvusManager({...})
results = milvus.search("查询", top_k=10, user_level=3)
for r in results:
    print(f"{r['metadata']['source']}: Level {r['kb_level']}")
```

### Q4: 如何修改已上传文档的权限级别？
**A**: 当前需要：
1. 删除旧文档（知识库管理 → 删除）
2. 重新上传并选择新级别

## 🔧 高级配置

### 自定义权限级别名称
修改 `ui/kb_manager.py` 中的映射：
```python
format_func=lambda x: {
    1: "🌐 公开级",
    2: "🔒 保密级",
    3: "🔐 绝密级",
    4: "🛡️ 核心机密级"  # 可扩展
}[x]
```

### 添加部门维度过滤
修改 Milvus 检索表达式：
```python
# 在 milvus_manager.py::search() 中
filter_expr = f"kb_level <= {user_level} AND department == '{user_department}'"
```

## 📚 相关文档
- 完整设计文档: `PERMISSION_SYSTEM_IMPLEMENTATION.md`
- 项目架构: `.github/copilot-instructions.md`
- Milvus 过滤语法: https://milvus.io/docs/boolean.md

---

**版本**: v1.0
**更新日期**: 2025-10-11
**作者**: AI BI Assistant Team
