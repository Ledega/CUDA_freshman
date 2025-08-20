#!/bin/bash

# build.sh - 增强版构建脚本

# 参数处理
TARGET=""
CLEAN_ONLY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --clean|-c)
            CLEAN_ONLY=true
            shift
            ;;
        --target|-t)
            TARGET="$2"
            shift 2
            ;;
        *)
            TARGET="$1"
            shift
            ;;
    esac
done

# 设置项目根目录
PROJECT_ROOT=$(pwd)
BUILD_DIR="${PROJECT_ROOT}/build"

# 创建build目录（如果不存在）
if [ ! -d "${BUILD_DIR}" ]; then
    mkdir "${BUILD_DIR}"
    echo "创建build目录"
fi

# 进入build目录
cd "${BUILD_DIR}"

# 清理旧的构建文件
echo "清理旧的构建文件..."
rm -rf *

# 如果只清理，则退出
if [ "$CLEAN_ONLY" = true ]; then
    echo "只清理完成"
    exit 0
fi

# 运行cmake
echo "运行CMake配置..."
cmake ..

# 检查cmake是否成功
if [ $? -ne 0 ]; then
    echo "CMake配置失败"
    exit 1
fi

# 构建项目
if [ -z "$TARGET" ]; then
    echo "开始完整构建..."
    make -j$(nproc)
else
    echo "构建目标: $TARGET"
    make -j$(nproc) "$TARGET"
fi

# 检查构建是否成功
if [ $? -ne 0 ]; then
    echo "构建失败"
    exit 1
fi

echo "构建完成！"