# Conda 核心命令速览
Conda 是跨平台的包管理与环境管理工具，适用于 Python/R 等编程语言，可隔离不同项目的依赖环境。以下是日常开发高频使用的核心命令：

## 一、环境管理
| 命令 | 功能 | 示例 |
|------|------|------|
| `conda create -n <env_name> python=<version>` | 创建新环境 | `conda create -n my_env python=3.9` |
| `conda activate <env_name>` | 激活环境 | `conda activate ml_env` |
| `conda deactivate` | 退出当前环境 | - |
| `conda env list` / `conda info --envs` | 列出所有环境 | - |
| `conda remove -n <env_name> --all` | 删除指定环境 | `conda remove -n my_env --all` |
| `conda env export > environment.yml` | 导出环境配置 | - |
| `conda env create -f environment.yml` | 从配置文件创建环境 | - |

## 二、包管理
| 命令 | 功能 | 示例 |
|------|------|------|
| `conda install <package>` | 安装包（当前环境） | `conda install pandas=1.5` |
| `conda install -n <env_name> <package>` | 安装包到指定环境 | `conda install -n ml_env scikit-learn` |
| `conda update <package>` | 更新包 | `conda update numpy` |
| `conda remove <package>` | 卸载包 | `conda remove pandas` |
| `conda list` | 列出当前环境已安装包 | - |
| `conda search <package>` | 搜索可安装的包版本 | `conda search tensorflow` |

## 三、基础配置与维护
| 命令 | 功能 | 示例 |
|------|------|------|
| `conda update conda` | 更新 conda 本身 | - |
| `conda clean -p -y` | 清理未使用的包缓存 | - |
| `conda config --add channels <channel>` | 添加镜像源 | `conda config --add channels conda-forge` |
| `conda config --show-sources` | 查看配置源 | - |

## 核心说明
1. 环境名建议简洁且与项目关联（如 `data_analysis`、`torch_env`）；
2. 指定 Python 版本可避免依赖兼容问题；
3. `conda-forge` 是常用第三方镜像源，包含更多包版本；
4. 导出的 `environment.yml` 可分享给他人，快速复现相同环境。