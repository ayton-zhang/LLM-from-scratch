# [git_workflow] branch_merge_workflow 代码解释笔记

---

## 原始代码

```bash
# 1. 创建并切换到新分支
git checkout -b feature/test-code-review

# 等价于：
git branch feature/test-code-review
git checkout feature/test-code-review

# 2. 提交改动
git add part_2/test_code_review.py
git commit -m "add test file with bare except issue for CodeRabbit review"

# 3. Push 到远程
git push origin feature/test-code-review

# 4. 合并回 main
git checkout main
git pull origin main
git merge feature/test-code-review
git push origin main
```

---

## 语法用法

### `git checkout -b <branch-name>`
- `-b` 表示“创建一个新分支并立即切换到该分支”。
- 等价于先运行 `git branch`（创建指针）再运行 `git checkout`（移动 HEAD）。
- 新版 Git（2.23+）推荐使用 `git switch -c <branch-name>` 替代。

### `git add <file>`
- 将文件改动放入**暂存区（staging area / index）**。
- Git 的提交是两步：先 `add`（暂存），再 `commit`（正式记录）。
- 用 `git add .` 可以暂存所有改动，但建议精确指定文件，避免意外提交无关内容。

### `git commit -m "<message>"`
- 将暂存区内容打包成一个 commit 节点，并附上提交信息。
- 每个 commit 有唯一的 SHA-1 哈希值（如 `a3f8c2d`），是 Git 追踪历史的基本单位。
- `-m` 后的信息应简洁描述"做了什么"，如 `fix: bare except clause in load_checkpoint`。

### `git push origin <branch-name>`
- 将本地分支的 commit 推送到远程仓库（`origin` 是远程仓库的默认别名）。
- 首次 push 可加 `-u`（`--set-upstream`），之后就可以直接用 `git push`：
  ```bash
  git push -u origin feature/test-code-review
  ```

### `git pull origin main`
- 等价于 `git fetch origin main` + `git merge origin/main`。
- 在 merge 之前先 pull，是为了把远程 main 的最新 commit 同步到本地，减少冲突风险。

### `git merge <branch-name>`
- 将指定分支的改动合并到当前分支。
- 默认产生一个 **merge commit**（三方合并），保留完整历史；如果没有分叉则触发 **fast-forward**（直接移动指针，不产生额外节点）。

---

## 代码逻辑

### 步骤 1：创建新分支

```
main:    A → B → C
                  ↑
              HEAD（当前位置）
```

运行 `git checkout -b feature/test-code-review` 后，Git 在 C 这个 commit 上新建一个指针 `feature/test-code-review`，并把 HEAD 指向它。此时两个分支完全相同，只是多了一个标签。

```
main:    A → B → C
                  ↑
feature:         (同一个 C，HEAD 在此)
```

### 步骤 2：提交改动

在新分支上 add + commit，产生新节点 D：

```
main:    A → B → C
                  \
feature:           D   ← HEAD
```

main 的指针没有动，只有 feature 分支向前推进。

### 步骤 3：Push 到远程

把本地的 D 节点上传到远程仓库，让 GitHub/GitLab 上也有 `feature/test-code-review` 分支，方便 CodeRabbit 等工具进行 code review。

### 步骤 4：合并回 main

**情况一 — Fast-forward（main 无新提交）**

```
merge 前：
main:    A → B → C
feature:           D

merge 后（main 指针直接前移，无额外节点）：
main:    A → B → C → D
```

**情况二 — Three-way merge（main 有新提交 F）**

```
merge 前：
main:    A → B → C → F
feature:           D

merge 后（生成 merge commit M）：
main:    A → B → C → F → M
                   ↘       ↗
feature:            D
```

Git 找到公共祖先 C，对比 C→F 和 C→D 的差异，自动合并。若两者修改了同一行则产生 conflict，需手动解决后再 commit。

---

## 约定俗成规范（分支命名与 PR 流程）

### 为什么 push 分支后会看到 Pull Request 提示
- 你执行 `git push origin test-code` 后，远程仓库会发现这是一个相对 `main` 有差异的新分支。
- GitHub 会据此给出“Compare & pull request”提示，目的是引导你进入团队协作流程，而不是自动合并。
- 这个提示来自托管平台工作流（GitHub/GitLab），不是 Git 本体命令的一部分。

### Pull Request 的协作价值
- **代码评审（Code Review）**：让他人在合并前检查逻辑、风格、边界条件。
- **自动检查（CI）**：在合并前跑测试、lint、静态检查、CodeRabbit 评论。
- **讨论与审计记录**：PR 会保存“改了什么、为什么改、谁审批”的完整记录。
- **受保护分支策略**：很多团队禁止直接 push 到 `main`，必须通过 PR 合并。

### `feature/` 前缀为什么常见
- `feature/` 不是 Git 语法强制要求，而是团队命名约定（convention）。
- 主要价值是“可读性和可管理性”：一眼区分这是新功能分支，而不是修复或运维分支。
- 也方便 CI/CD 按前缀匹配规则，例如只对 `feature/*` 触发某些检查。

### 常见分支前缀建议
- `feature/<name>`：新功能开发
- `fix/<name>`：普通缺陷修复
- `hotfix/<name>`：线上紧急修复
- `chore/<name>`：工程杂务（依赖升级、脚本、重构清理）
- `docs/<name>`：文档更新
- `test/<name>`：测试相关改动

### 实操建议
- 个人实验可以用简单名（如 `test-code`），团队协作建议使用前缀（如 `feature/code-rabbit-review-test`）。
- 分支名尽量短且语义清晰，统一用小写加连字符。
- 合并前先同步主分支（`git pull origin main`），可提前发现冲突并在本地解决。

---

## 关键知识点总结

1. **Branch 是轻量级指针**：Git 中的分支本质上只是一个指向某个 commit 的文件（`.git/refs/heads/<name>`），创建分支几乎零成本，不会复制任何代码。

2. **Three-way merge 基于公共祖先**：合并时 Git 不是直接比较两个分支的最新状态，而是先找到它们的**最近公共祖先（Lowest Common Ancestor）**，再分别对比双方的改动，决定如何合并或标记冲突。

3. **先 pull 再 merge 是最佳实践**：在 merge feature 分支之前先 `git pull origin main`，确保本地 main 与远程同步，可以将冲突提前暴露在本地，而不是推送后才发现。
