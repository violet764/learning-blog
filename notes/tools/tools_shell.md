# Shell 

## Shell 基础与文件系统

### Shell 概述

- Shell 是**文字接口**，允许执行程序并获取结构化输出
- 本课程使用 **Bourne Again Shell (bash)**
- Shell 是**编程环境**，具备变量、条件、循环和函数

**终端与提示符**
```bash
missing:~$ 
```

提示符格式解析：
- `missing`：主机名
- `~`：当前工作目录（home）
- `$`：非root用户标识（root用户为 `#`）

### 基本命令使用

**命令执行**
```bash
# 执行程序
missing:~$ date
Fri 10 Jan 2020 11:49:31 AM EST

# 带参数执行
missing:~$ echo hello
hello
```

**环境变量 PATH**
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

### 文件系统导航

**路径概念**
- **绝对路径**：以 `/` 开头的完整路径
- **相对路径**：相对于当前工作目录的路径
- **特殊路径符号**：
  - `.`：当前目录
  - `..`：上级目录
  - `~`：用户主目录

**导航命令**
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

**文件列表与权限**
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

**常用文件操作命令**
- `mv`：重命名或移动文件
- `cp`：拷贝文件
- `mkdir`：新建文件夹
- `rm`：删除文件
- `man`：查看程序手册（使用 `q` 退出）

### 程序间连接

**输入输出重定向**
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

**管道 (Pipes)**
```bash
# 管道连接程序输入输出
missing:~$ ls -l / | tail -n1
drwxr-xr-x 1 root  root  4096 Jun 20  2019 var

# 复杂管道示例
missing:~$ curl --head --silent google.com | grep --ignore-case content-length | cut --delimiter=' ' -f2
219
```

**管道机制**：将一个命令的输出直接作为另一个命令的输入

### 用户权限与系统管理

Linux 将用户分为三类，构成权限控制的核心维度：
| 主体类型 | 标识 | 说明 |
|----------|------|------|
| 所有者（User） | `u` | 文件/目录的创建者，通常拥有最高操作权限 |
| 所属组（Group） | `g` | 所属用户组的成员，可共享组内权限 |
| 其他用户（Other） | `o` | 系统中除所有者、所属组外的所有用户 |
| 所有用户（All） | `a` | 包含 `u`+`g`+`o`（默认省略时等价于 `a`） |


针对文件和目录，Linux 定义了三种基础操作权限：
| 权限标识 | 字符表示 | 数字表示 | 文件含义 | 目录含义 |
|----------|----------|----------|----------|----------|
| 读权限 | `r` | `4` | 可查看文件内容（如 `cat`/`less`） | 可列出目录内文件（如 `ls`） |
| 写权限 | `w` | `2` | 可修改/删除文件内容（如 `vim`/`rm`） | 可创建/删除/重命名目录内文件（如 `touch`/`rm`） |
| 执行权限 | `x` | `1` | 可将文件作为程序/脚本执行（如 `./script.sh`） | 可进入目录（如 `cd`） |
| 无权限 | `-` | `0` | 无对应操作权限 | 无对应操作权限 |

> 关键区别：  
> - 目录的 `w` 权限≠修改目录本身，而是修改目录内的文件列表；  
> - 目录的 `x` 权限是基础：无 `x` 则无法进入目录，即使有 `r` 也无法查看内容。

修改权限
```bash
chmod [选项] 权限规则 目标文件/目录

# 给文件所有者添加执行权限
chmod u+x script.sh

# 移除所属组的写权限
chmod g-w test.txt

# 给所有用户设置读+执行权限（重置）
chmod a=rx app.bin

# 最常用：所有者全权限，组和其他读+执行（755）
chmod 755 /usr/local/bin/myapp

# 仅所有者可读写执行，组和其他无权限（700）
chmod 700 ~/.ssh/id_rsa


```
**Root 用户和 sudo**
- **Root 用户**：特殊用户，几乎不受限制
- 通常不直接以`root`身份登录
- `sudo` 命令：以`super user`身份执行操作
- 遇到"`permission denied`"错误时可能需要使用`sudo`

**系统配置 (sysfs)**
```bash
# 查找亮度文件
$ sudo find -L /sys/class/backlight -maxdepth 2 -name '*brightness*'

# 修改亮度（正确方式）
$ echo 3 | sudo tee brightness

# 修改LED状态
$ echo 1 | sudo tee /sys/class/leds/input6::scrolllock/brightness
```

> **注意**：系统参数暴露在 `/sys` 下，Linux可直接修改内核参数
---

## Shell 脚本编程

### 变量与字符串

**变量定义与访问**
```bash
# 变量赋值（注意：无空格）
foo=bar

# 变量访问
echo "$foo"  # 输出: bar（变量替换）
echo '$foo'  # 输出: $foo（原义字符串）
```

**特殊变量**
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

### 控制流与条件判断

**条件运算符**
```bash
# 短路运算符
false || echo "Oops, fail"    # 输出: Oops, fail
true && echo "Things went well"  # 输出: Things went well
false ; echo "This will always run"  # 总是执行
```

**函数定义**
```bash
# 函数定义示例
mcd () {
    mkdir -p "$1"
    cd "$1"
}
```

### 命令替换与进程替换

**命令替换**
```bash
# 传统命令替换
for file in $(ls)

# 进程替换（更安全）
diff <(ls foo) <(ls bar)
```

**完整脚本示例**
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

###  Shell 通配符与模式匹配

**基本通配符**
```bash
rm foo?    # 删除 foo1, foo2 等（单字符）
rm foo*    # 删除 foo 开头的所有文件
```

**花括号展开**
```bash
convert image.{png,jpg}  # 展开为 convert image.png image.jpg
cp /path/to/project/{foo,bar,baz}.sh /newpath

# 结合使用
mv *{.py,.sh} folder
touch {foo,bar}/{a..h}
```

### Shell 函数与脚本的区别

| 特性 | 函数 | 脚本 |
|------|------|------|
| 语言 | 只能使用shell语言 | 可使用任意语言 |
| 加载 | 定义时加载 | 每次执行时加载 |
| 执行环境 | 当前shell环境 | 单独进程 |
| 环境变量 | 可直接修改 | 通过export传递 |

---

## Shell 工具与高级命令

### 查看命令用法

**获取帮助的方式**
- `man <command>`：显示详细手册
- `<command> -h` 或 `<command> --help`：显示帮助信息
- `tldr <command>`：显示简化的使用示例

### 文件查找工具

**find 命令**
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

**现代替代工具**
- **fd**：更简单、快速的find替代品
- **locate**：基于数据库的快速文件名搜索

### 代码搜索工具

**grep 使用**
```bash
grep -C 5 pattern      # 显示前后5行上下文
grep -v pattern         # 反选，显示不匹配的行
grep -R pattern         # 递归搜索目录
```

**`ripgrep (rg)` - 现代替代**
```bash
rg -t py 'import requests'      # 搜索Python文件
rg -u --files-without-match "^#!"  # 查找没有shebang的文件
rg foo -A 5                    # 显示匹配后5行
rg --stats PATTERN             # 显示统计信息
```

### 命令历史与导航

**历史命令管理**
- `history`：显示命令历史
- `Ctrl+R`：历史命令搜索
- **fzf**：模糊查找工具
- 历史自动补全功能

**文件夹导航工具**
- **fasd**：基于频率和时效的文件/目录跳转
- **autojump**：智能目录跳转
- **tree**：目录树显示
- **broot**, **nnn**, **ranger**：文件管理器

---

## 高级 Shell 技能与实用命令

### 文本处理高级工具

**sed - 流编辑器**
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

**awk - 文本处理工具**
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

**cut - 列提取工具**
```bash
cut -d: -f1,6 /etc/passwd      # 提取第1和6列，使用冒号分隔
cut -c1-10 file.txt            # 提取每行的1-10字符
cut -f2 -d' ' file.txt         # 提取第2列，使用空格分隔
```

**sort 和 uniq**
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

### 进程管理与系统监控

**进程管理命令**
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

**系统资源监控**
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

### 网络工具与诊断

**网络连接与测试**
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

**文件传输工具**
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

### SSH 与远程连接

**SSH 基础使用**
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

**SSH 配置文件**
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

### Shell 环境配置与定制

**配置文件**
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

**环境变量**
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

### 高级文本处理技巧

**正则表达式基础**
```bash
# grep正则表达式
grep '^start' file            # 以start开头的行
grep 'end$' file              # 以end结尾的行
grep '[0-9]\+' file           # 包含一个或多个数字
grep 'word1\|word2' file      # 匹配word1或word2
grep '(word1).*word2' file    # word1后面有word2的行
```

**多文件处理技巧**
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

## Shell 最佳实践与安全

###  脚本安全实践

**输入验证与安全**
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

**错误处理**
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

### 性能优化技巧

**并行处理**
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

**资源管理**
```bash
# 限制资源使用
ulimit -v 1048576  # 限制虚拟内存使用
ulimit -t 60       # 限制CPU时间

# 内存使用监控
free -m
ps aux --sort=-%mem | head
```

### 常用实用命令

**时间与日期**
```bash
date                          # 当前时间
date +"%Y-%m-%d %H:%M:%S"     # 格式化时间
date -d "2020-01-01" +%s      # 转换为时间戳
date -d @1577836800           # 时间戳转换为日期
cal                           # 显示日历
```

**杂项实用命令**
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

---

## Linux 基础命令速查表
**一、文件系统操作命令**
| 命令 | 基本语法 | 核心功能 | 常用选项/示例 | 补充说明 |
|------|----------|----------|--------------|----------|
| `pwd` | `pwd [选项]` | 显示当前工作目录的绝对路径 | `-P`：显示物理路径（忽略软链接）<br>`pwd` → `/home/user/docs` | 无参数时默认显示逻辑路径 |
| `cd` | `cd [目录路径]` | 切换工作目录 | `cd ~`：回到用户主目录<br>`cd ..`：回到上级目录<br>`cd -`：回到上一次所在目录 | 无参数时等价于 `cd ~` |
| `ls` | `ls [选项] [目录/文件]` | 列出目录内容或文件信息 | `-l`：详细列表（权限、大小、时间等）<br>`-a`：显示隐藏文件（以.开头）<br>`-h`：人类可读大小（如1K/2M）<br>`ls -lh /home` | 无参数时列出当前目录文件 |
| `mkdir` | `mkdir [选项] 目录名` | 创建新目录 | `-p`：递归创建多级目录<br>`-m`：指定权限（如mkdir -m 755 test）<br>`mkdir -p dir1/dir2` | 默认权限由umask决定（通常755） |
| `rmdir` | `rmdir [选项] 目录名` | 删除空目录 | `-p`：递归删除空父目录<br>`rmdir -p dir1/dir2` | 仅能删除空目录，非空需用`rm -r` |
| `touch` | `touch [选项] 文件名` | 创建空文件/更新文件时间戳 | `-t`：指定时间戳（如touch -t 202501011200 file）<br>`touch newfile.txt` | 若文件已存在，仅更新访问/修改时间 |
| `cp` | `cp [选项] 源 目标` | 复制文件/目录 | `-r/-R`：递归复制目录<br>`-i`：覆盖前提示<br>`-p`：保留文件属性（权限、时间）<br>`cp -r dir1 dir2` | 复制目录必须加`-r`选项 |
| `mv` | `mv [选项] 源 目标` | 移动/重命名文件/目录 | `-i`：覆盖前提示<br>`-f`：强制覆盖（忽略提示）<br>`mv old.txt new.txt`（重命名）<br>`mv file /tmp/`（移动） | 跨文件系统移动时实际是复制+删除 |
| `rm` | `rm [选项] 文件/目录` | 删除文件/目录 | `-r/-R`：递归删除目录<br>`-f`：强制删除（无提示）<br>`-i`：删除前提示<br>`rm -rf dir1` | `rm -rf /` 是高危操作，禁止执行！ |
| `ln` | `ln [选项] 源 链接名` | 创建链接文件 | `-s`：创建软链接（符号链接）<br>`ln -s /etc/passwd passwd.link`（软链接）<br>`ln file hardlink`（硬链接） | 硬链接不能跨文件系统，软链接可以 |

**二、文件内容查看命令**
| 命令 | 基本语法 | 核心功能 | 常用选项/示例 | 补充说明 |
|------|----------|----------|--------------|----------|
| `cat` | `cat [选项] 文件` | 正序显示文件全部内容 | `-n`：显示行号<br>`-b`：仅显示非空行号<br>`cat -n test.txt` | 适合查看小文件，大文件推荐`less` |
| `tac` | `tac [选项] 文件` | 倒序显示文件内容 | `tac test.txt` | 与`cat`相反，从最后一行开始输出 |
| `more` | `more [选项] 文件` | 分页显示文件内容 | 空格键：下一页<br>Enter：下一行<br>`q`：退出<br>`more -5 test.txt`（每页5行） | 仅能向下翻页，无法向上 |
| `less` | `less [选项] 文件` | 交互式分页显示文件内容 | `↑/↓`：上下翻行<br>`PageUp/PageDown`：上下翻页<br>`/关键词`：搜索<br>`q`：退出<br>`less test.txt` | 支持上下翻页，比`more`更灵活 |
| `head` | `head [选项] 文件` | 显示文件开头部分内容 | `-n 数字`：指定显示行数（默认10行）<br>`head -5 test.txt` | `head -n -5` 显示除最后5行外的内容 |
| `tail` | `tail [选项] 文件` | 显示文件末尾部分内容 | `-n 数字`：指定显示行数（默认10行）<br>`-f`：实时跟踪文件更新<br>`tail -f /var/log/syslog` | `tail -n +5` 从第5行开始显示全部内容 |
| `wc` | `wc [选项] 文件` | 统计文件行数/单词数/字节数 | `-l`：仅统计行数<br>`-w`：仅统计单词数<br>`-c`：仅统计字节数<br>`wc -l test.txt` | 无选项时输出 行数 单词数 字节数 文件名 |

**三、权限与用户管理命令**
| 命令 | 基本语法 | 核心功能 | 常用选项/示例 | 补充说明 |
|------|----------|----------|--------------|----------|
| `chmod` | `chmod [选项] 权限 目标` | 修改文件/目录权限 | `-R`：递归修改<br>`chmod 755 script.sh`（数字法）<br>`chmod u+x test.sh`（符号法） | 仅所有者/root可修改权限 |
| `chown` | `chown [选项] 属主:属组 目标` | 修改文件/目录的所有者/所属组 | `-R`：递归修改<br>`chown root:root test.txt`<br>`chown user: test.txt`（仅改属主） | 仅root可修改所有者 |
| `chgrp` | `chgrp [选项] 属组 目标` | 修改文件/目录的所属组 | `-R`：递归修改<br>`chgrp users test.txt` | 属主可将文件改到自己所属的组 |
| `whoami` | `whoami` | 显示当前登录用户名 | `whoami` → `user` | 无选项，简单快速查看当前用户 |
| `who` | `who [选项]` | 显示当前登录系统的所有用户 | `-u`：显示用户空闲时间<br>`who` → `user tty1 2025-01-01 10:00` | 包含用户名、终端、登录时间等 |
| `id` | `id [选项] [用户名]` | 显示用户UID/GID及所属组 | `-u`：仅显示UID<br>`-g`：仅显示GID<br>`id root` | 无参数时显示当前用户信息 |
| `su` | `su [选项] [用户名]` | 切换用户身份 | `-`：切换后加载目标用户环境<br>`su - root`（切换到root）<br>`su user`（仅切换身份，不加载环境） | 无用户名时默认切换到root |
| `sudo` | `sudo [选项] 命令` | 以root权限执行命令 | `-i`：切换到root交互式shell<br>`sudo ls /root` | 需要配置/etc/sudoers文件授权 |

**四、系统信息与进程管理命令**
| 命令 | 基本语法 | 核心功能 | 常用选项/示例 | 补充说明 |
|------|----------|----------|--------------|----------|
| `uname` | `uname [选项]` | 显示系统内核/硬件信息 | `-a`：显示所有信息<br>`-r`：显示内核版本<br>`uname -a` | 无选项时仅显示系统名称（如Linux） |
| `top` | `top [选项]` | 实时监控系统进程与资源占用 | `P`：按CPU排序<br>`M`：按内存排序<br>`k`：终止进程<br>`q`：退出 | 动态刷新，默认3秒一次 |
| `ps` | `ps [选项]` | 显示当前进程快照 | `ps aux`：显示所有进程（BSD风格）<br>`ps -ef`：显示所有进程（System V风格）<br>`ps aux | grep nginx` | 静态显示，需重新执行刷新 |
| `kill` | `kill [选项] 进程ID` | 向进程发送信号（默认终止进程） | `-9`：强制终止（SIGKILL）<br>`-15`：优雅终止（SIGTERM，默认）<br>`kill -9 1234` | 需知道进程PID，可通过`ps`查询 |
| `df` | `df [选项]` | 显示文件系统磁盘空间使用情况 | `-h`：人类可读大小<br>`-T`：显示文件系统类型<br>`df -h` | 无选项时以KB为单位显示 |
| `du` | `du [选项] 目录/文件` | 统计目录/文件的磁盘占用空间 | `-h`：人类可读大小<br>`-s`：仅显示总计<br>`du -sh /home` | 无参数时递归显示目录内各文件大小 |
| `free` | `free [选项]` | 显示系统内存使用情况 | `-h`：人类可读大小<br>`-m`：以MB为单位<br>`free -h` | 包含总内存、已用、空闲、缓存等信息 |
| `date` | `date [选项] [格式]` | 显示/设置系统时间 | `date +"%Y-%m-%d %H:%M:%S"` → `2025-01-01 12:00:00`<br>`date -s "2025-01-01 12:00"`（设置时间） | 需root权限设置时间 |

**五、网络操作命令**
| 命令 | 基本语法 | 核心功能 | 常用选项/示例 | 补充说明 |
|------|----------|----------|--------------|----------|
| `ping` | `ping [选项] 主机/IP` | 测试网络连通性 | `-c 次数`：指定ping的次数<br>`ping -c 5 baidu.com` | 按Ctrl+C终止，默认无限次 |
| `ifconfig` | `ifconfig [网卡名] [选项]` | 配置/显示网络接口信息 | `ifconfig eth0`（显示eth0信息）<br>`ifconfig eth0 192.168.1.100`（设置IP） | 部分系统已被`ip`命令替代 |
| `ip` | `ip [子命令] [选项]` | 新一代网络配置工具 | `ip addr`（显示IP地址）<br>`ip link set eth0 up`（启用网卡）<br>`ip route`（显示路由表） | 功能覆盖ifconfig/route等命令 |
| `netstat` | `netstat [选项]` | 显示网络连接/端口/路由等信息 | `-t`：显示TCP连接<br>`-u`：显示UDP连接<br>`-l`：显示监听端口<br>`netstat -tuln` | 部分系统推荐使用`ss`命令替代 |
| `curl` | `curl [选项] URL` | 发送HTTP/HTTPS请求，下载文件 | `-O`：保存文件（保留原名）<br>`-L`：跟随重定向<br>`curl -O https://example.com/file.txt` | 可测试接口、下载资源 |
| `wget` | `wget [选项] URL` | 从网络下载文件 | `-O 文件名`：指定保存名称<br>`-c`：断点续传<br>`wget -c https://example.com/file.txt` | 后台下载，适合大文件 |
| `ssh` | `ssh [选项] 用户@主机` | 远程登录Linux服务器 | `-p 端口`：指定SSH端口<br>`ssh -p 2222 root@192.168.1.1` | 默认端口22，需目标服务器开启SSH服务 |

**六、压缩与解压命令**
| 命令 | 基本语法 | 核心功能 | 常用选项/示例 | 补充说明 |
|------|----------|----------|--------------|----------|
| `tar` | `tar [选项] 归档文件 源文件/目录` | 打包/解压文件（支持压缩） | `-c`：创建归档<br>`-x`：解压归档<br>`-z`：gzip压缩<br>`-j`：bzip2压缩<br>`-f`：指定归档文件名<br>`tar -czf test.tar.gz dir/`（打包并gzip压缩）<br>`tar -xzf test.tar.gz`（解压gzip压缩包） | 最常用的打包压缩工具，兼容多种格式 |
| `gzip` | `gzip [选项] 文件` | 压缩文件（生成.gz文件） | `-d`：解压.gz文件<br>`-k`：保留原文件<br>`gzip test.txt`（压缩为test.txt.gz）<br>`gzip -d test.txt.gz`（解压） | 仅能压缩单个文件，目录需先tar打包 |
| `unzip` | `unzip [选项] 压缩包` | 解压.zip格式压缩包 | `-d 目录`：指定解压目录<br>`unzip test.zip -d /tmp` | 需提前安装unzip工具（yum install unzip） |


