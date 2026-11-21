# MIGraphX 自定义预处理算子示例 — Makefile
# 来源：金山文档《MIGRAPHX 自定义预处理算子（C++）实现流程深度解析》§6.8
# 编译验证：hipcc + /opt/dtk 头文件/库（海光 DCU 用 hipcc，非 g++/dcc）

# 编译器（海光 DCU 用 hipcc，来自 DTK）
CC = hipcc
# C++ 标准
CXXFLAGS = -std=c++17 -g
# 头文件路径（DTK 头文件 + OpenCV）
INCLUDES = -I/opt/dtk/include -I/usr/include/opencv4
# 库路径（DTK 库）
LDFLAGS = -L/opt/dtk/lib
# 链接库（MIGraphX 系列 + OpenCV + pthread/stdc++）
LIBS = -Wl,--start-group -lmigraphx_onnx -lmigraphx_gpu -lmigraphx_ref -lmigraphx -lmigraphx_device -lmigraphx_c -Wl,--end-group -lpthread -lstdc++ -lopencv_core -lopencv_imgcodecs -lopencv_imgproc

# 可执行文件名称
TARGET = resnet50_custom_op
# 源文件列表
SRCS = main.cpp op_add.cpp resnet50_preprocess.cpp
# 把 .cpp 替换成 .o，生成目标文件列表（增量编译核心）
OBJS = $(SRCS:.cpp=.o)

# 默认目标：编译可执行文件
all: $(TARGET)

# 链接：只把更新过的 .o 文件链接成可执行文件
$(TARGET): $(OBJS)
	$(CC) $(CXXFLAGS) $(INCLUDES) $(LDFLAGS) -o $@ $^ $(LIBS)

# 编译：单个 .cpp 文件 → .o 文件（只有修改过的 .cpp 才会重新编译）
%.o: %.cpp
	$(CC) $(CXXFLAGS) $(INCLUDES) -c -o $@ $<

# 清理编译产物
clean:
	rm -f $(TARGET) $(OBJS)

# 伪目标（避免和同名文件冲突）
.PHONY: all clean
