# 🚀 Deep-Learner: 基于 Agentic RAG 的启发式 AI 导师

**Deep-Learner** 是一款专为技术领域（如 Spark、数据科学）设计的智能学习辅助系统。本项目跳出了传统的线性 RAG 模式，采用 **Agentic Workflow** 架构，实现了“规划-检索-教学-审计”的完整闭环。

---

## 🌟 核心工程亮点

### 1. 工业级路径脱敏与溯源协议 (Logical Citation Mapping)
针对生产环境下 RAG 系统的安全性痛点，本项目自研了一套隔离映射机制：
* **物理-逻辑隔离**：`Retriever` 节点通过 `os.path.basename` 自动屏蔽本地绝对路径，防止服务器文件系统结构泄露。
* **逻辑 ID 映射**：在 `AgentState` 中维护 `source_mapping` 字典（ID -> 纯净文件名），模型仅处理逻辑 ID（如 [[资料 1]]），显著降低 Token 消耗并提升引用准确率。

### 2. 基于 LangGraph 的自省架构 (Self-Reflection)
本项目利用 LangGraph 的状态机特性，构建了具备自我修正能力的 Agent：
* **逻辑审计 (Critic Node)**：引入专门的审计节点，对生成内容进行**忠实度（Faithfulness）**与**幻觉检测**。
* **动态重规划 (Re-planning)**：当审计未通过时，系统会自动触发路由跳转回 `Planner` 重新检索，确保输出结果的严谨性。

### 3. 全链路可观察性 (Observability)
* **结构化日志**：使用自定义的 `AgentStep` 协议记录每个节点的思考链（Chain of Thought）。
* **状态持久化**：利用 Python `Annotated` 语法实现非破坏性的状态更新，完整保留推理痕迹，方便生产环境下的调试与溯源。

---

## 🏗️ 架构全景图



- **LLM 编排**: LangGraph
- **向量数据库**: Milvus
- **全文检索**: Elasticsearch (Hybrid Search)
- **前端交互**: Chainlit (支持侧边栏原文预览)

---

## 🛠️ 技术栈

| 组件 | 选型 |
| :--- | :--- |
| **语言** | Python 3.9+ |
| **Agent 框架** | LangGraph (State-centric design) |
| **基础模型** | OpenAI / Claude (通过 Factory 模式接入) |
| **UI 框架** | Chainlit |

---

## 🚀 快速启动

### 1. 安装依赖
```bash
pip install -e .
```
### 2. 环境配置
在根目录创建 `.env` 文件并填入必要的 API Key：

## 3. 启动服务
```bash
chainlit run interface/app.py -w
```
### 👨‍💻 关于作者
关于本项目由 JHU 学生 Justin Jiao 开发，专注于 AI Agent 应用工程化实践。如有任何问题或建议，欢迎通过 GitHub 提交 Issue 或 Pull Request。
## 联系方式
- GitHub: [justinjiao](https://github.com/justinjiao)
- Email: tuoshengjiao@gamil.com
---
*感谢您的关注，祝您使用愉快！*