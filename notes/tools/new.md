# Git 底层原理与数据模型核心总结
## 一、核心设计思想
1. **定位**：分布式版本控制标准，解决文件变更追踪与多人协作问题
2. **核心差异**：
   - 区别于 SVN 等集中式 VCS，本地仓库包含完整历史，无网络可正常操作
   - 采用快照式存储（而非差异存储），未修改文件通过哈希引用复用
3. **核心特性**：
   - 数据不可变：提交记录创建后无法修改，修改本质是新建对象
   - 哈希校验：所有数据通过 SHA-1 哈希（40 位十六进制）唯一标识，确保完整性
   - 自底向上理解：先掌握数据模型，再理解命令逻辑，无需死记硬背

## 二、三大核心对象（Objects）
Git 底层通过不可变对象存储数据，均存储于 `.git/objects/` 目录
| 对象类型 | 作用 | 核心特性 | 伪代码定义 |
|----------|------|----------|------------|
| Blob（数据对象） | 存储文件原始内容 | 仅存数据，不含文件名/权限；相同内容自动去重 | `type blob = array<byte>` |
| Tree（目录树对象） | 存储目录结构 | 映射文件名到 Blob/子 Tree；记录文件权限（如 100644 普通文件、040000 目录） | `type tree = map<string, {type: "blob"|"tree", hash: string, mode: string}>` |
| Commit（提交对象） | 存储提交元信息 | 关联顶级 Tree 快照；通过父引用形成历史链 | `type commit = struct { parents: array<string>, author: string, committer: string, message: string, snapshot: string }` |

## 三、引用（References）：哈希的友好别名
用于解决哈希难记忆问题，存储于 `.git/refs/` 目录，本质是可变指针
1. **分支引用（refs/heads/）**：指向分支最新提交，如 `refs/heads/main`
2. **标签引用（refs/tags/）**：固定重要提交（如版本 v1.0），分轻量标签和附注标签
3. **HEAD 引用**：指向当前所在分支/提交，是定位"当前位置"的核心

## 四、版本历史：有向无环图（DAG）
1. **模型本质**：通过 Commit 父引用构建的 DAG，而非线性历史
2. **核心优势**：支持多分支并行开发，合并冲突本地解决，完整保留分支轨迹
3. **结构示例**：
```plaintext
o (commit: c1) <-- o (commit: c2) <-- o (commit: c3)  # main 分支
                    ^
                     \
                      o (commit: c4) <-- o (commit: c5)  # feature 分支
```

- 分岔：基于 c2 创建 feature 分支，c3 与 c4 并行开发
- 合并：创建新提交 c6，父引用为 c3 和 c5，形成合并节点

## 五、暂存区（Index）：提交中间层
1. **作用**：缓存待提交的修改，支持"部分提交"，优化提交性能
2. **工作流程**：工作区修改 → `git add` → 暂存区 → `git commit` → 生成 Tree/Commit
3. **特性**：可通过 `git reset HEAD <file>` 撤销暂存，未 add 的修改不纳入提交

## 六、核心操作底层映射
| 日常命令 | 底层逻辑 |
|----------|----------|
| `git add <file>` | 生成 Blob 对象，更新暂存区关联 |
| `git commit -m "msg"` | 基于暂存区生成 Tree，创建 Commit，更新分支引用 |
| `git branch <name>` | 复制当前 Commit 哈希到新分支引用文件 |
| `git checkout <branch>` | 更新 HEAD 指向，检出对应 Tree 到工作区 |
| `git merge <branch>` | 找共同祖先，对比 Tree 差异，创建合并 Commit（多父引用） |
| `git reset --hard <hash>` | 更新分支引用，覆盖工作区/暂存区为目标 Commit 快照 |

## 七、仓库核心目录（.git/）

```plaintext
.git/
├── objects/          # 存储所有 Blob/Tree/Commit 对象（按哈希前缀分目录）
├── refs/             # 存储引用：分支（heads/）、标签（tags/）、远程分支（remotes/）
├── HEAD              # 指向当前分支/提交
├── index             # 暂存区数据
├── config            # 仓库配置（用户信息、远程仓库、别名等）
└── logs/             # 引用变更日志（用于 git reflog 恢复操作）
```


## 八、关键启示与实践原则
1. **数据安全**：`git reset --hard` 会覆盖本地修改，需谨慎使用（可通过 `git reflog` 恢复）
2. **分支特性**：分支创建仅复制指针，成本极低，鼓励多分支开发
3. **协作原则**：已推送的提交不修改历史，需撤销变更用 `git revert`
4. **性能优化**：相同文件复用 Blob 对象，快照存储高效节省空间

## 九、推荐学习资源
1. Pro Git 中文版：深入讲解数据模型与底层实现
2. Git for Computer Scientists：图解 Git 数据模型
3. Learn Git Branching：交互式游戏学习分支与合并
4. Oh Shit, Git!?!：快速解决 Git 操作失误