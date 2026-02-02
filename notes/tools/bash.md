# Bash 脚本语法详解

## 脚本基础

### Shebang 行
每个bash脚本都应该以shebang行开始，指定解释器路径：

```bash
#!/bin/bash
# 或者使用更通用的方式
#!/usr/bin/env bash
```

### 注释
```bash
# 单行注释

: '这是
   多行
   注释'

# 或者使用 << 语法
<<COMMENT
这是另一种
多行注释
COMMENT
```

### 脚本执行方式
```bash
# 方式1：直接执行（需要执行权限）
chmod +x script.sh
./script.sh

# 方式2：通过bash解释器执行
bash script.sh
sh script.sh

# 方式3：在当前shell中执行
source script.sh
. script.sh
```

## 变量

### 变量定义与使用
```bash
# 变量定义（等号前后不能有空格）
name="John"
age=25
files_count=$(ls | wc -l)

# 变量使用
echo "Name: $name"
echo "Age: ${age}"  # 推荐使用花括号

# 命令替换
current_date=$(date)
file_list=`ls *.txt`  # 传统方式，不推荐
```

### 特殊变量
```bash
echo "脚本名: $0"           # 脚本名称
echo "第一个参数: $1"       # 位置参数
echo "所有参数: $@"         # 所有参数列表
echo "参数个数: $#"         # 参数数量
echo "退出状态: $?"         # 上一条命令的退出状态
echo "进程ID: $$"           # 当前脚本的进程ID
echo "后台进程ID: $!"       # 最后一个后台进程的ID
```

### 变量操作
```bash
name="John Doe"

# 字符串长度
echo "长度: ${#name}"

# 子字符串提取
echo "前4字符: ${name:0:4}"
echo "从第5开始: ${name:5}"

# 字符串替换
echo "替换: ${name/John/Jane}"
echo "全局替换: ${name//o/O}"

# 默认值设置
unset undefined_var
echo "默认值: ${undefined_var:-默认值}"
echo "设置默认: ${undefined_var:=默认值}"
```

## 数据类型

### 字符串
```bash
str1="Hello"
str2='World'
str3=$str1$str2           # 字符串拼接
str4="${str1}, ${str2}!"  # 带空格拼接

# 字符串比较
if [[ "$str1" == "Hello" ]]; then
    echo "字符串相等"
fi

# 空字符串检查
if [[ -z "$str1" ]]; then
    echo "字符串为空"
fi
```

### 数字
```bash
# 整数运算
num1=10
num2=3

# 算术运算
echo $((num1 + num2))     # 加法
echo $((num1 - num2))     # 减法
echo $((num1 * num2))     # 乘法
echo $((num1 / num2))     # 除法（整数）
echo $((num1 % num2))     # 取模

# 复合运算
((num1++))                # 自增
((num2 += 5))             # 复合赋值

# 浮点数运算（需要bc）
echo "scale=2; 10/3" | bc
```

### 数组
```bash
# 数组定义
fruits=("apple" "banana" "orange")
numbers=(1 2 3 4 5)

# 数组访问
echo "第一个元素: ${fruits[0]}"
echo "所有元素: ${fruits[@]}"
echo "元素个数: ${#fruits[@]}"

# 数组遍历
for fruit in "${fruits[@]}"; do
    echo "水果: $fruit"
done

# 关联数组（bash 4.0+）
declare -A user_info
user_info["name"]="John"
user_info["age"]="25"
echo "姓名: ${user_info[name]}"
```

## 流程控制

### 条件判断

#### if 语句
```bash
# 基本语法
if [[ condition ]]; then
    # 代码块
elif [[ another_condition ]]; then
    # 代码块
else
    # 代码块
fi

# 文件测试
if [[ -f "file.txt" ]]; then
    echo "文件存在"
fi

if [[ -d "/path/to/dir" ]]; then
    echo "目录存在"
fi

if [[ -r "file.txt" ]]; then
    echo "文件可读"
fi

# 字符串比较
if [[ "$str1" == "$str2" ]]; then
    echo "字符串相等"
fi

if [[ "$str1" < "$str2" ]]; then
    echo "str1 在字典序上小于 str2"
fi

# 数值比较
if [[ $num1 -eq $num2 ]]; then
    echo "相等"
fi

if [[ $num1 -gt $num2 ]]; then
    echo "大于"
fi
```

#### case 语句
```bash
read -p "输入选择 (1-3): " choice

case $choice in
    1)
        echo "选择了选项1"
        ;;
    2)
        echo "选择了选项2"
        ;;
    3)
        echo "选择了选项3"
        ;;
    *)
        echo "无效选择"
        ;;
esac

# 模式匹配
case $filename in
    *.txt)
        echo "文本文件"
        ;;
    *.jpg|*.png)
        echo "图片文件"
        ;;
    *)
        echo "其他文件"
        ;;
esac
```

### 循环

#### for 循环
```bash
# 遍历列表
for fruit in apple banana orange; do
    echo "水果: $fruit"
done

# 遍历数组
for file in "${files[@]}"; do
    echo "处理文件: $file"
done

# C风格for循环
for ((i=0; i<10; i++)); do
    echo "数字: $i"
done

# 遍历文件
for file in *.txt; do
    echo "文本文件: $file"
done
```

#### while 循环
```bash
# 基本while循环
count=1
while [[ $count -le 5 ]]; do
    echo "计数: $count"
    ((count++))
done

# 读取文件行
while IFS= read -r line; do
    echo "行内容: $line"
done < "file.txt"

# 无限循环
while true; do
    echo "正在运行..."
    sleep 1
done
```

#### until 循环
```bash
# 直到条件为真
count=10
until [[ $count -eq 0 ]]; do
    echo "倒计时: $count"
    ((count--))
done
```

### 循环控制
```bash
# break - 退出循环
for i in {1..10}; do
    if [[ $i -eq 5 ]]; then
        break
    fi
    echo "数字: $i"
done

# continue - 跳过当前迭代
for i in {1..10}; do
    if [[ $i -eq 5 ]]; then
        continue
    fi
    echo "数字: $i"
done
```

## 函数

### 函数定义与调用
```bash
# 函数定义
function say_hello() {
    local name="$1"  # 局部变量
    echo "Hello, $name!"
    return 0
}

# 简写方式
say_goodbye() {
    echo "Goodbye, $1!"
}

# 函数调用
say_hello "John"
result=$(say_goodbye "Jane")
```

### 函数参数
```bash
process_files() {
    local file1="$1"
    local file2="$2"
    local verbose="${3:-false}"  # 默认参数
    
    echo "处理文件: $file1 和 $file2"
    if [[ "$verbose" == "true" ]]; then
        echo "详细模式已启用"
    fi
    
    return 0
}

process_files "a.txt" "b.txt" "true"
```

### 返回值处理
```bash
# 返回数值
add_numbers() {
    local result=$(( $1 + $2 ))
    return $result
}

add_numbers 5 3
echo "加法结果: $?"

# 返回字符串
get_info() {
    local info="系统时间: $(date)"
    echo "$info"  # 通过stdout返回
}

system_info=$(get_info)
echo "$system_info"
```

## 输入输出

### 标准输入输出
```bash
# 输出重定向
echo "Hello" > output.txt        # 覆盖写入
echo "World" >> output.txt       # 追加写入

# 输入重定向
cat < input.txt
wc -l < file.txt

# 错误重定向
command 2> error.log             # 错误输出到文件
command 2>&1                     # 错误重定向到标准输出
command > output.log 2>&1        # 标准输出和错误都重定向
```

### 管道
```bash
# 基本管道
ls -la | grep "txt"
cat file.txt | sort | uniq

# 进程替换
diff <(sort file1.txt) <(sort file2.txt)
```

### 用户输入
```bash
# 基本输入
read -p "请输入姓名: " name
echo "你好, $name"

# 带超时的输入
read -t 10 -p "10秒内输入: " input

# 静默输入（密码）
read -s -p "输入密码: " password
echo

# 读取多个值
read -p "输入姓名和年龄: " name age
echo "姓名: $name, 年龄: $age"

# 从文件读取
while IFS= read -r line; do
    echo "行: $line"
done < "file.txt"
```

## 高级特性

### 字符串操作
```bash
str="Hello World"

# 大小写转换（bash 4.0+）
echo "大写: ${str^^}"
echo "小写: ${str,,}"

# 首字母大小写
echo "首字母大写: ${str^}"

# 字符串长度
echo "长度: ${#str}"

# 子字符串
echo "第2-5字符: ${str:1:4}"

# 模式匹配
echo "匹配World: ${str#* }"     # 删除最短匹配前缀
echo "匹配Hello: ${str% *}"     # 删除最短匹配后缀
```

### 正则表达式
```bash
# 使用 [[ =~ ]] 进行正则匹配
email="test@example.com"

if [[ "$email" =~ ^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$ ]]; then
    echo "有效的邮箱地址"
else
    echo "无效的邮箱地址"
fi

# 提取匹配组
if [[ "$str" =~ (Hello).*(World) ]]; then
    echo "组1: ${BASH_REMATCH[1]}"
    echo "组2: ${BASH_REMATCH[2]}"
fi
```

### 调试和错误处理
```bash
# 调试模式
set -x  # 开启调试
# 脚本代码
set +x  # 关闭调试

# 严格模式
set -euo pipefail
# -e: 遇到错误立即退出
# -u: 使用未定义变量时报错
# -o pipefail: 管道中任一命令失败则整个管道失败

# 错误处理
trap 'echo "错误发生在第 $LINENO 行"' ERR

# 清理处理
trap 'rm -f /tmp/tempfile; echo "清理完成"' EXIT

# 自定义错误处理
handle_error() {
    echo "错误代码: $?"
    echo "错误命令: $BASH_COMMAND"
    echo "错误行号: $LINENO"
    exit 1
}

trap handle_error ERR
```

### 信号处理
```bash
# 处理Ctrl+C (SIGINT)
trap 'echo "\n程序被中断"; exit 1' INT

# 处理终止信号 (SIGTERM)
trap 'echo "收到终止信号"; cleanup; exit 0' TERM

# 忽略信号
trap '' HUP  # 忽略挂起信号

# 恢复默认处理
trap - INT   # 恢复默认的Ctrl+C处理
```

## 实用脚本示例

### 文件备份脚本
```bash
#!/bin/bash

# 严格模式
set -euo pipefail

# 配置变量
BACKUP_DIR="/backup"
SOURCE_DIR="$1"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="backup_${TIMESTAMP}.tar.gz"

# 参数检查
if [[ $# -eq 0 ]]; then
    echo "用法: $0 <源目录>"
    exit 1
fi

if [[ ! -d "$SOURCE_DIR" ]]; then
    echo "错误: 目录 $SOURCE_DIR 不存在"
    exit 1
fi

# 创建备份目录
mkdir -p "$BACKUP_DIR"

# 执行备份
echo "开始备份 $SOURCE_DIR..."
tar -czf "$BACKUP_DIR/$BACKUP_NAME" "$SOURCE_DIR"

# 检查备份是否成功
if [[ $? -eq 0 ]]; then
    echo "备份成功: $BACKUP_DIR/$BACKUP_NAME"
    
    # 清理7天前的备份
    find "$BACKUP_DIR" -name "backup_*.tar.gz" -mtime +7 -delete
    echo "已清理7天前的备份文件"
else
    echo "备份失败"
    exit 1
fi
```

### 系统监控脚本
```bash
#!/bin/bash

# 监控系统资源
monitor_system() {
    local threshold=80
    
    # CPU使用率
    local cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    
    # 内存使用率
    local mem_info=$(free | grep Mem)
    local mem_total=$(echo "$mem_info" | awk '{print $2}')
    local mem_used=$(echo "$mem_info" | awk '{print $3}')
    local mem_usage=$((mem_used * 100 / mem_total))
    
    # 磁盘使用率
    local disk_usage=$(df / | awk 'NR==2 {print $5}' | cut -d'%' -f1)
    
    echo "=== 系统监控报告 ==="
    echo "CPU使用率: ${cpu_usage}%"
    echo "内存使用率: ${mem_usage}%"
    echo "根分区使用率: ${disk_usage}%"
    
    # 告警检查
    if [[ $cpu_usage -gt $threshold ]]; then
        echo "⚠️  CPU使用率过高!"
    fi
    
    if [[ $mem_usage -gt $threshold ]]; then
        echo "⚠️  内存使用率过高!"
    fi
    
    if [[ $disk_usage -gt $threshold ]]; then
        echo "⚠️  磁盘使用率过高!"
    fi
}

# 主循环
while true; do
    monitor_system
    echo "------------------------"
    sleep 60  # 每分钟检查一次
done
```

### 日志分析脚本
```bash
#!/bin/bash

analyze_logs() {
    local log_file="${1:-/var/log/syslog}"
    
    if [[ ! -f "$log_file" ]]; then
        echo "错误: 日志文件 $log_file 不存在"
        return 1
    fi
    
    echo "=== 日志分析报告 ==="
    echo "分析文件: $log_file"
    echo "文件大小: $(du -h "$log_file" | cut -f1)"
    echo "总行数: $(wc -l < "$log_file")"
    echo
    
    # 错误统计
    echo "错误统计:"
    grep -i "error" "$log_file" | wc -l
    
    # 警告统计
    echo "警告统计:"
    grep -i "warning" "$log_file" | wc -l
    
    # 最近10条错误
    echo "最近10条错误:"
    grep -i "error" "$log_file" | tail -10
}

# 使用函数
analyze_logs "$1"
```

## 最佳实践

### 代码风格
```bash
# 使用有意义的变量名
good: user_name="john"
bad: x="john"

# 使用双引号引用变量
good: echo "Hello, $name"
bad: echo Hello, $name

# 使用 [[ ]] 代替 [ ]
good: if [[ $name == "john" ]]; then
bad: if [ $name = "john" ]; then

# 使用函数封装代码
process_user() {
    local user="$1"
    # 处理逻辑
}
```

### 错误处理
```bash
# 总是检查命令执行状态
if ! command; then
    echo "命令执行失败"
    exit 1
fi

# 使用trap进行清理
trap 'rm -f /tmp/*.tmp' EXIT

# 提供有用的错误信息
echo "错误: 文件 $filename 不存在" >&2
```

### 性能优化
```bash
# 避免在循环中调用外部命令
# 不好:
for file in *; do
    size=$(wc -c < "$file")
    echo "$file: $size"
done

# 好:
for file in *; do
    echo -n "$file: "
    wc -c < "$file"
done
```

这个文档详细介绍了bash脚本的各种语法特性，从基础到高级，包含了大量实用的示例代码。你可以根据实际需求参考相应的语法部分。