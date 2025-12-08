# Shell 课程学习指南

## 内容概述

1. **Shell 基础**：Shell 概念、基本命令、文件系统导航和程序间连接
2. **Shell 工具与脚本**：Shell 编程、高级工具和最佳实践

文档采用循序渐进的学习结构，从基础概念到高级应用，适合初学者和有一定经验的学习者使用。

---

## 第一部分：Shell 基础与文件系统

### 1. Shell 概述

#### 什么是 Shell
- Shell 是**文字接口**，允许执行程序并获取结构化输出
- 本课程使用 **Bourne Again Shell (bash)**
- Shell 是**编程环境**，具备变量、条件、循环和函数

#### 终端与提示符
```bash
missing:~$ 
```

提示符格式解析：
- `missing`：主机名
- `~`：当前工作目录（home）
- `$`：非root用户标识（root用户为 `#`）

### 2. 基本命令使用

#### 命令执行
```bash
# 执行程序
missing:~$ date
Fri 10 Jan 2020 11:49:31 AM EST

# 带参数执行
missing:~$ echo hello
hello
```

#### 环境变量 PATH
```bash
# 查看PATH环境变量
missing:~$ echo $PATH
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# 查找程序位置
missing:~$ which echo
/bin/echo

# 直接指定路径执行
missing:~$ /bin/echo $PATH
```

**PATH 变量**：定义了系统查找可执行文件的路径顺序

### 3. 文件系统导航

#### 路径概念
- **绝对路径**：以 `/` 开头的完整路径
- **相对路径**：相对于当前工作目录的路径
- **特殊路径符号**：
  - `.`：当前目录
  - `..`：上级目录
  - `~`：用户主目录

#### 导航命令
```bash
# 查看当前目录
missing:~$ pwd
/home/missing

# 切换目录
missing:~$ cd /home
missing:/home$ cd ..
missing:/$ cd ./home
missing:/home$ cd missing
```

#### 文件列表与权限
```bash
# 列出文件
missing:~$ ls

# 详细列出文件信息
missing:~$ ls -l /home
drwxr-xr-x 1 missing  users  4096 Jun 15  2019 missing
```

**权限解释**：
- 第一个字符：`d`（目录）或 `-`（文件）
- 后续9字符分3组：所有者、用户组、其他用户
- 每组3字符：`r`（读）、`w`（写）、`x`（执行）

#### 常用文件操作命令
- `mv`：重命名或移动文件
- `cp`：拷贝文件
- `mkdir`：新建文件夹
- `rm`：删除文件
- `man`：查看程序手册（使用 `q` 退出）

### 4. 程序间连接

#### 输入输出重定向
```bash
# 输出重定向到文件（覆盖）
missing:~$ echo hello > hello.txt

# 从文件输入
missing:~$ cat < hello.txt

# 同时输入输出重定向
missing:~$ cat < hello.txt > hello2.txt

# 追加内容
missing:~$ echo hello >> file.txt
```

#### 管道 (Pipes)
```bash
# 管道连接程序输入输出
missing:~$ ls -l / | tail -n1
drwxr-xr-x 1 root  root  4096 Jun 20  2019 var

# 复杂管道示例
missing:~$ curl --head --silent google.com | grep --ignore-case content-length | cut --delimiter=' ' -f2
219
```

**管道机制**：将一个命令的输出直接作为另一个命令的输入

### 5. 用户权限与系统管理

#### Root 用户和 sudo
- **Root 用户**：特殊用户，几乎不受限制
- 通常不直接以root身份登录
- `sudo` 命令：以super user身份执行操作
- 遇到"permission denied"错误时可能需要使用sudo

#### 系统配置 (sysfs)
```bash
# 查找亮度文件
$ sudo find -L /sys/class/backlight -maxdepth 2 -name '*brightness*'

# 修改亮度（正确方式）
$ echo 3 | sudo tee brightness

# 修改LED状态
$ echo 1 | sudo tee /sys/class/leds/input6::scrolllock/brightness
```

**注意**：系统参数暴露在 `/sys` 下，可直接修改内核参数（Windows和macOS无此功能）

---

## 第二部分：Shell 脚本编程

### 1. 变量与字符串

#### 变量定义与访问
```bash
# 变量赋值（注意：无空格）
foo=bar

# 变量访问
echo "$foo"  # 输出: bar（变量替换）
echo '$foo'  # 输出: $foo（原义字符串）
```

#### 特殊变量
| 变量 | 含义 |
|------|------|
| `$0` | 脚本名 |
| `$1` 到 `$9` | 脚本参数 |
| `$@` | 所有参数 |
| `$#` | 参数个数 |
| `$?` | 前一个命令的返回值 |
| `$$` | 当前脚本的进程识别码 |
| `!!` | 完整的上一条命令 |
| `$_` | 上一条命令的最后一个参数 |

### 2. 控制流与条件判断

#### 条件运算符
```bash
# 短路运算符
false || echo "Oops, fail"    # 输出: Oops, fail
true && echo "Things went well"  # 输出: Things went well
false ; echo "This will always run"  # 总是执行
```

#### 函数定义
```bash
# 函数定义示例
mcd () {
    mkdir -p "$1"
    cd "$1"
}
```

### 3. 命令替换与进程替换

#### 命令替换
```bash
# 传统命令替换
for file in $(ls)

# 进程替换（更安全）
diff <(ls foo) <(ls bar)
```

#### 完整脚本示例
```bash
#!/bin/bash

echo "Starting program at $(date)"
echo "Running program $0 with $# arguments with pid $$"

for file in "$@"; do
    grep foobar "$file" > /dev/null 2> /dev/null
    if [[ $? -ne 0 ]]; then
        echo "File $file does not have any foobar, adding one"
        echo "# foobar" >> "$file"
    fi
done
```

### 4. Shell 通配符与模式匹配

#### 基本通配符
```bash
rm foo?    # 删除 foo1, foo2 等（单字符）
rm foo*    # 删除 foo 开头的所有文件
```

#### 花括号展开
```bash
convert image.{png,jpg}  # 展开为 convert image.png image.jpg
cp /path/to/project/{foo,bar,baz}.sh /newpath

# 结合使用
mv *{.py,.sh} folder
touch {foo,bar}/{a..h}
```

### 5. Shell 函数与脚本的区别

| 特性 | 函数 | 脚本 |
|------|------|------|
| 语言 | 只能使用shell语言 | 可使用任意语言 |
| 加载 | 定义时加载 | 每次执行时加载 |
| 执行环境 | 当前shell环境 | 单独进程 |
| 环境变量 | 可直接修改 | 通过export传递 |

---

## 第三部分：Shell 工具与高级命令

### 1. 查看命令用法

#### 获取帮助的方式
- `man <command>`：显示详细手册
- `<command> -h` 或 `<command> --help`：显示帮助信息
- `tldr <command>`：显示简化的使用示例

### 2. 文件查找工具

#### find 命令
```bash
# 查找所有名称为src的文件夹
find . -name src -type d

# 查找路径中包含test的python文件
find . -path '*/test/*.py' -type f

# 查找前一天修改的文件
find . -mtime -1

# 查找特定大小的文件
find . -size +500k -size -10M -name '*.tar.gz'

# 执行操作
find . -name '*.tmp' -exec rm {} \;
find . -name '*.png' -exec magick {} {}.jpg \;
```

#### 现代替代工具
- **fd**：更简单、快速的find替代品
- **locate**：基于数据库的快速文件名搜索

### 3. 代码搜索工具

#### grep 使用
```bash
grep -C 5 pattern      # 显示前后5行上下文
grep -v pattern         # 反选，显示不匹配的行
grep -R pattern         # 递归搜索目录
```

#### ripgrep (rg) - 现代替代
```bash
rg -t py 'import requests'      # 搜索Python文件
rg -u --files-without-match "^#!"  # 查找没有shebang的文件
rg foo -A 5                    # 显示匹配后5行
rg --stats PATTERN             # 显示统计信息
```

### 4. 命令历史与导航

#### 历史命令管理
- `history`：显示命令历史
- `Ctrl+R`：历史命令搜索
- **fzf**：模糊查找工具
- 历史自动补全功能

#### 文件夹导航工具
- **fasd**：基于频率和时效的文件/目录跳转
- **autojump**：智能目录跳转
- **tree**：目录树显示
- **broot**, **nnn**, **ranger**：文件管理器

---

## 第四部分：高级 Shell 技能与实用命令

### 1. 文本处理高级工具

#### sed - 流编辑器
```bash
# 基本替换
sed 's/old/new/g' file.txt    # 替换所有匹配项
sed -i 's/old/new/g' file.txt # 直接修改文件

# 删除操作
sed '3d' file.txt             # 删除第3行
sed '/pattern/d' file.txt     # 删除匹配模式的行

# 打印特定行
sed -n '1,5p' file.txt        # 打印1-5行
sed -n '/pattern/p' file.txt   # 打印匹配模式的行

# 复杂操作
sed -e 's/foo/bar/' -e 's/baz/qux/' file.txt  # 多个编辑命令
```

#### awk - 文本处理工具
```bash
# 基本用法
awk '{print $1}' file.txt      # 打印每行第一列
awk '{print $NF}' file.txt     # 打印每行最后一列

# 条件处理
awk '$3 > 100 {print $1, $2}' data.txt
awk '/error/ {count++} END {print count}' log.txt

# 自定义分隔符
awk -F: '{print $1, $6}' /etc/passwd

# 统计计算
awk '{sum += $3} END {print "Average:", sum/NR}' data.txt
```

#### cut - 列提取工具
```bash
cut -d: -f1,6 /etc/passwd      # 提取第1和6列，使用冒号分隔
cut -c1-10 file.txt            # 提取每行的1-10字符
cut -f2 -d' ' file.txt         # 提取第2列，使用空格分隔
```

#### sort 和 uniq
```bash
# 排序命令
sort file.txt                  # 按字母顺序排序
sort -n file.txt              # 按数值顺序排序
sort -r file.txt              # 逆序排序
sort -k2 file.txt             # 按第2列排序

# 去重命令
uniq file.txt                  # 去除相邻重复行
sort file.txt | uniq           # 去除所有重复行
sort file.txt | uniq -c        # 统计重复次数
```

### 2. 进程管理与系统监控

#### 进程管理命令
```bash
# 查看进程
ps aux                        # 查看所有进程详细信息
ps -ef                        # 另一种查看进程的格式
top                           # 实时进程监控
htop                          # 增强版top（需要安装）

# 进程控制
kill PID                      # 终止进程
kill -9 PID                   # 强制终止进程
killall process_name          # 按名称终止进程
pkill pattern                 # 按模式匹配终止进程

# 后台任务
command &                     # 后台运行命令
jobs                          # 查看后台任务
fg %1                         # 将任务1调到前台
bg %1                         # 将任务1调到后台
nohup command &               # 忽略挂起信号，后台运行
```

#### 系统资源监控
```bash
# 系统信息
uname -a                      # 系统详细信息
df -h                         # 磁盘使用情况（人类可读）
du -sh /path/to/dir           # 目录大小统计
free -h                       # 内存使用情况

# 资源监控
iostat                        # I/O统计
vmstat                        # 虚拟内存统计
lscpu                         # CPU信息
lsblk                         # 块设备信息
```

### 3. 网络工具与诊断

#### 网络连接与测试
```bash
# 网络测试
ping google.com               # 测试网络连通性
traceroute google.com         # 跟踪网络路径
mtr google.com                # 结合ping和traceroute的工具

# 端口和服务
netstat -tuln                 # 查看监听端口
ss -tuln                      # 现代替代netstat
lsof -i :80                   # 查看端口80占用情况
telnet host port              # 测试端口连通性

# 网络配置
ip addr show                  # 显示网络接口
ifconfig                      # 传统网络配置命令
dig google.com                # DNS查询
nslookup google.com           # DNS查询（传统）
```

#### 文件传输工具
```bash
# 文件传输
scp file.txt user@host:/path/ # 安全复制文件
rsync -av src/ dest/          # 高效文件同步
wget https://example.com/file # 下载文件
curl -O https://example.com/file # 下载文件（保留原名）

# 压缩与解压
tar -czf archive.tar.gz dir/  # 创建gzip压缩包
tar -xzf archive.tar.gz       # 解压gzip压缩包
zip -r archive.zip dir/       # 创建zip文件
unzip archive.zip             # 解压zip文件
```

### 4. SSH 与远程连接

#### SSH 基础使用
```bash
# 基本连接
ssh user@hostname             # 基本SSH连接
ssh -p 2222 user@hostname     # 指定端口连接
ssh user@hostname command     # 在远程主机执行命令

# 密钥管理
ssh-keygen -t rsa -b 4096     # 生成RSA密钥对
ssh-copy-id user@hostname     # 复制公钥到远程主机
ssh-add                       # 添加私钥到ssh代理

# SSH隧道
ssh -L 8080:localhost:80 user@host  # 本地端口转发
ssh -R 8080:localhost:80 user@host  # 远程端口转发
ssh -D 1080 user@hostname          # 动态端口转发（SOCKS代理）
```

#### SSH 配置文件
```bash
# ~/.ssh/config 配置示例
Host server1
    HostName example.com
    User username
    Port 22
    IdentityFile ~/.ssh/id_rsa

Host *
    Compression yes
    ServerAliveInterval 60
```

### 5. Shell 环境配置与定制

#### 配置文件
```bash
# ~/.bashrc - bash交互式shell配置
export PATH="$PATH:/usr/local/bin"
export EDITOR="vim"
export HISTSIZE=10000
export HISTCONTROL=ignoredups

# 别名定义
alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'
alias grep='grep --color=auto'
alias ..='cd ..'
alias ...='cd ../..'

# 自定义函数
extract() {
    if [ -f "$1" ]; then
        case "$1" in
            *.tar.bz2) tar xjf "$1" ;;
            *.tar.gz)  tar xzf "$1" ;;
            *.zip)     unzip "$1" ;;
            *)         echo "'$1' cannot be extracted via extract()" ;;
        esac
    fi
}
```

#### 环境变量
```bash
# 常用环境变量
export PATH                   # 可执行文件搜索路径
export HOME                   # 用户主目录
export USER                   # 当前用户名
export PWD                    # 当前工作目录
export LANG                   # 语言设置
export PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '  # 自定义提示符

# 会话级环境变量
export TEMP_DIR="/tmp/mytemp"
export LOG_LEVEL="DEBUG"
```

### 6. 高级文本处理技巧

#### 正则表达式基础
```bash
# grep正则表达式
grep '^start' file            # 以start开头的行
grep 'end$' file              # 以end结尾的行
grep '[0-9]\+' file           # 包含一个或多个数字
grep 'word1\|word2' file      # 匹配word1或word2
grep '(word1).*word2' file    # word1后面有word2的行
```

#### 多文件处理技巧
```bash
# 批量重命名
for file in *.txt; do
    mv "$file" "${file%.txt}.bak"
done

# 批量处理
for dir in */; do
    (cd "$dir" && ls -la > ../"${dir%/}_files.txt")
done

# find结合exec
find . -name "*.log" -exec rm {} \;
find . -name "*.sh" -exec chmod +x {} \;
```

---

## 第五部分：Shell 最佳实践与安全

### 1. 脚本安全实践

#### 输入验证与安全
```bash
# 参数验证
if [ $# -eq 0 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# 文件存在性检查
if [ ! -f "$1" ]; then
    echo "Error: File $1 does not exist"
    exit 1
fi

# 安全的文件处理
filename="$(basename "$1")"
if [[ "$filename" =~ [^a-zA-Z0-9._-] ]]; then
    echo "Error: Unsafe filename"
    exit 1
fi
```

#### 错误处理
```bash
# 严格模式
set -euo pipefail  # 遇到错误退出，未定义变量报错，管道错误传播

# 错误捕获
trap 'echo "Error occurred at line $LINENO"' ERR

# 检查命令执行状态
if command; then
    echo "Success"
else
    echo "Failed with exit code $?"
fi
```

### 2. 性能优化技巧

#### 并行处理
```bash
# 使用xargs并行
find . -name "*.jpg" | xargs -P 4 -I {} convert {} {}.png

# 后台进程并行
for i in {1..10}; do
    process_task $i &
done
wait  # 等待所有后台任务完成

# GNU parallel
find . -name "*.txt" | parallel process_file
```

#### 资源管理
```bash
# 限制资源使用
ulimit -v 1048576  # 限制虚拟内存使用
ulimit -t 60       # 限制CPU时间

# 内存使用监控
free -m
ps aux --sort=-%mem | head
```

### 3. 常用实用命令

#### 时间与日期
```bash
date                          # 当前时间
date +"%Y-%m-%d %H:%M:%S"     # 格式化时间
date -d "2020-01-01" +%s      # 转换为时间戳
date -d @1577836800           # 时间戳转换为日期
cal                           # 显示日历
```

#### 杂项实用命令
```bash
# 随机数
echo $RANDOM                  # 0-32767随机数
shuf -i 1-100 -n 5           # 1-100随机数选5个

# 数学计算
echo $((2+2))                 # 算术运算
echo "scale=4; 3/2" | bc      # 高精度计算

# 字符串处理
string="Hello, World"
echo ${#string}               # 字符串长度
echo ${string:0:5}            # 子字符串
echo ${string//World/Shell}   # 字符串替换

# 历史命令
history | grep command        # 搜索历史命令
!number                       # 执行历史命令号number
!!                            # 执行上一条命令
```
