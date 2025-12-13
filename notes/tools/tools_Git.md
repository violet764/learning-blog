# Git 核心笔记：底层原理与常用命令

## Git 底层核心原理

**Git 设计思想与数据结构**

**快照式版本控制**：Git 基于**快照**存储文件完整状态，而非传统系统的**差异**存储。

**核心优势**：
- 数据完整性：SHA-1 哈希校验保证不可篡改
- 分布式架构：每个本地仓库都是完整版本库
- 存储高效：未修改文件引用原快照，避免重复

**4种核心对象**（均以 SHA-1 哈希标识）：
| 对象 | 作用 | 特点 |
|------|------|------|
| **Blob** | 存储文件内容 | 不含文件名，仅数据 |
| **Tree** | 存储目录结构 | 记录文件/子目录哈希与权限 |
| **Commit** | 存储版本信息 | 关联 tree + 父 commit + 作者/信息 |
| **Tag** | 标记重要版本 | 如 v1.0 版本标签 |

**历史模型与引用系统**

**版本历史（有向无环图）**：
```
o (c1) <-- o (c2) <-- o (c3)     # main 分支
          ^
           \
            o (c4) <-- o (c5)     # feature 分支
```

**引用系统**（解决哈希记忆问题）：
- **分支引用**（`refs/heads/`）：指向分支最新提交
- **标签引用**（`refs/tags/`）：固定重要版本
- **HEAD 引用**：指向当前分支/提交

**核心优势**：支持多分支并行开发、完整保留历史轨迹。


**工作流与操作映射**

**工作区域与目录结构**：
```
.git/
├── objects/       # 存储所有对象
├── refs/          # 引用目录
├── HEAD           # 指向当前分支
├── index          # 暂存区文件
└── config         # 仓库配置
```

**4个工作区域**：
1. **工作区**：本地编辑文件
2. **暂存区**：待提交快照（`git add` 后进入）
3. **本地仓库**：`.git` 目录存储所有版本
4. **远程仓库**：GitHub/GitLab 等协作平台

**工作流程**：
```
工作区 →(add)→ 暂存区 →(commit)→ 本地仓库 →(push)→ 远程仓库
远程仓库 →(pull)→ 本地仓库 →(checkout)→ 工作区
```

**核心操作底层映射**：
| 命令 | 底层逻辑 |
|------|----------|
| `git add` | 生成 Blob，更新暂存区关联 |
| `git commit` | 基于暂存区生成 Tree，创建 Commit |
| `git branch` | 复制当前 Commit 哈希到新分支引用 |
| `git checkout` | 更新 HEAD，检出对应 Tree 到工作区 |
| `git merge` | 找共同祖先，对比差异，创建合并 Commit |

## Git 常用命令

### 仓库初始化与配置

**基础配置**

```Bash
# 全局配置（所有仓库生效）
git config --global user.name "你的用户名"
git config --global user.email "你的邮箱"
git config --global core.editor "vim"  # 设置默认编辑器
git config --global core.autocrlf false  # 关闭换行符自动转换（跨平台建议）

# 查看配置
git config --list  # 查看所有配置
git config user.name  # 查看指定配置

# 初始化本地仓库
git init  # 在当前目录创建.git仓库
git init 仓库名  # 创建指定名称的仓库目录并初始化
```

**克隆远程仓库**

```Bash
git clone <远程仓库URL>  # 完整克隆
git clone <URL> -b 分支名  # 克隆指定分支
git clone <URL> --depth 1  # 浅克隆（仅最新版本，节省空间）
```

### 文件操作

**状态查看**

```Bash
git status  # 查看工作区/暂存区状态
git status -s  # 简洁输出（推荐）
```

**添加/撤销暂存**

```Bash
git add 文件名  # 添加单个文件到暂存区
git add .  # 添加所有修改/新增文件到暂存区
git add -u  # 仅添加已跟踪文件的修改/删除（不含新增）

# 撤销暂存区文件（回到工作区）
git reset HEAD 文件名  # 撤销单个文件
git reset HEAD .  # 撤销所有暂存文件
```

**提交版本**

```Bash
git commit -m "提交说明"  # 暂存区文件提交到本地仓库
git commit -am "提交说明"  # 跳过暂存区，直接提交已跟踪文件的修改
git commit --amend  # 修正最后一次提交（如补全说明、添加漏文件）
```

**撤销工作区修改**

```Bash
# 恢复已跟踪文件到最近一次提交的状态（谨慎：会覆盖本地修改）
git checkout -- 文件名
```

### 分支操作

**分支查看**

```Bash
git branch  # 查看本地分支（*表示当前分支）
git branch -r  # 查看远程分支
git branch -a  # 查看所有分支（本地+远程）
```

**分支创建/切换/删除**

```Bash
git branch 分支名  # 创建分支（不切换）
git checkout 分支名  # 切换分支
git checkout -b 分支名  # 创建并切换分支（常用）
git switch 分支名  # Git 2.23+ 新增，替代 checkout 切换分支
git switch -c 分支名  # 创建并切换分支

# 删除分支
git branch -d 分支名  # 删除已合并的本地分支
git branch -D 分支名  # 强制删除未合并的本地分支
git push 远程名 --delete 分支名  # 删除远程分支
```

**分支合并**

```Bash
git merge 分支名  # 将指定分支合并到当前分支（如main合并feature分支）
git merge --no-ff 分支名  # 禁用快进合并（保留分支历史，推荐）

# 解决合并冲突
# 冲突文件会标记 <<<<<<< HEAD（当前分支）、=======（合并分支）、>>>>>>> 分支名
# 手动编辑冲突文件后，执行：
git add 冲突文件名
git commit -m "解决合并冲突：xxx"
```

**分支变基**

```Bash
git checkout feature  # 切换到功能分支
git rebase main  # 将feature分支基于main最新版本重放（使历史线性）
git rebase --abort  # 放弃变基
git rebase --continue  # 解决冲突后继续变基
```

### 版本回溯与查看

**提交历史查看**

```Bash
git log  # 查看提交历史（按时间倒序）
git log --oneline  # 简洁输出（哈希前缀+提交说明）
git log --graph  # 图形化显示分支合并历史
git log --author="用户名"  # 查看指定作者的提交
git log --since="2025-01-01"  # 查看指定时间后的提交
git log 文件名  # 查看单个文件的提交历史
```

**版本回溯**

```Bash
git reset --hard 提交哈希  # 回溯到指定版本（工作区/暂存区/本地仓库全重置）
git reset --soft 提交哈希  # 仅重置本地仓库，暂存区/工作区保留修改
git reset --mixed 提交哈希  # 重置本地仓库+暂存区，工作区保留（默认）

# 找回被reset的版本（Git会保留30天未引用的对象）
git reflog  # 查看所有操作记录（找到丢失的提交哈希）
git reset --hard 丢失的哈希  # 恢复版本
```

### 远程仓库交互

**远程仓库管理**

```Bash
git remote -v  # 查看远程仓库地址（fetch/push）
git remote add 远程名 <URL>  # 添加远程仓库（如origin）
git remote rename 旧名 新名  # 重命名远程仓库
git remote remove 远程名  # 删除远程仓库
git remote update  # 更新远程分支信息
```

**推送/拉取代码**

```Bash
git push 远程名 本地分支:远程分支  # 推送本地分支到远程
git push -u 远程名 分支名  # 关联本地分支与远程分支（首次推送用）
git push  # 已关联分支可直接推送

git pull 远程名 远程分支:本地分支  # 拉取远程分支并合并到本地分支
git pull  # 已关联分支可直接拉取（= git fetch + git merge）

git fetch 远程名  # 仅拉取远程仓库更新（不合并，可先查看差异）
git fetch 远程名 分支名  # 拉取指定远程分支
```

### 标签操作

```Bash
git tag  # 查看所有标签
git tag v1.0  # 给最新提交打轻量标签
git tag -a v1.0 -m "版本1.0"  # 打附注标签（含详细信息，推荐）
git tag -a v1.0 提交哈希 -m "版本1.0"  # 给指定提交打标签

git push 远程名 v1.0  # 推送单个标签到远程
git push 远程名 --tags  # 推送所有标签到远程

git checkout v1.0  # 切换到标签版本（只读，建议创建分支修改）
git tag -d v1.0  # 删除本地标签
git push 远程名 --delete v1.0  # 删除远程标签
```

### 进阶操作

**暂存工作区（stash）**

```Bash
git stash  # 暂存当前工作区/暂存区的修改（保留干净工作区）
git stash save "备注"  # 带备注的暂存
git stash list  # 查看暂存列表
git stash apply stash@{0}  # 应用指定暂存（不删除暂存记录）
git stash pop  # 应用最新暂存并删除记录（常用）
git stash drop stash@{0}  # 删除指定暂存
git stash clear  # 清空所有暂存
```

**忽略文件（.gitignore）**

创建 `.gitignore` 文件，写入需忽略的文件/目录：

```Plain Text
# 忽略所有.log文件
*.log
# 忽略node_modules目录
node_modules/
# 忽略特定文件
.env
# 例外：不忽略test.log
!test.log
# 忽略目录下所有文件，但保留目录结构
temp/*
!temp/.gitkeep
```

**查看差异**

```Bash
git diff  # 查看工作区与暂存区的差异
git diff --cached  # 查看暂存区与本地仓库的差异
git diff 分支1 分支2  # 查看两个分支的差异
git diff 提交哈希1 提交哈希2  # 查看两个提交的差异
```

## Git 实践

**核心原则**

1. **数据完整性**：所有操作基于哈希校验，确保版本不可篡改；

2. **分布式**：本地仓库包含完整历史，无网络也可提交；

3. **分支轻量化**：分支本质是指向commit的指针，创建/切换成本极低；

4. **尽量不修改历史**：已推送到远程的提交，避免用`reset --hard`/`rebase`修改（会导致协作冲突）。

**最佳实践**

1. 提交粒度：一个提交对应一个功能/修复，提交说明清晰（如「fix: 修复登录验证逻辑」）；

2. 分支规范：

    - `main/master`：主分支（稳定版本，仅合并已测试代码）；

    - `develop`：开发分支（日常开发，从main创建）；

    - `feature/xxx`：功能分支（从develop创建，完成后合并回develop）；

    - `bugfix/xxx`：修复分支（从main/develop创建，修复后合并）；

3. 定期拉取：协作时及时`git pull`，避免大量冲突；

4. 慎用`git clean`：`git clean -fd`会删除未跟踪文件/目录（不可恢复）；

5. 重要操作前备份：如变基、重置前，可先创建临时分支备份。

## 常见问题

**提交后发现漏文件**

```Bash
git add 漏加的文件
git commit --amend --no-edit  # 补充文件到最后一次提交，不修改说明
```

**误删分支恢复**

```Bash
git reflog  # 找到分支最后一次提交的哈希
git checkout -b 恢复的分支名 哈希值
```

**撤销已推送的提交**

```Bash
# 方法1：创建新提交撤销（推荐，不修改历史）
git revert 提交哈希
git push 远程名 分支名

# 方法2：强制推送（谨慎！会覆盖远程历史，仅个人分支使用）
git reset --hard 目标哈希
git push -f 远程名 分支名
```

