# LLM Subtitle Optimizer

使用大语言模型 (LLM) 优化和修正字幕内容，支持 **Agent Loop** 自动验证和修正。

## 核心特性

- **Agent Loop**: LLM → 验证 → 反馈 → 重试（最多 3 轮），确保输出质量
- **并发批量处理**: 支持多线程并行处理，大幅提升速度
- **自动对齐修复**: 处理优化过程中可能产生的段落合并或拆分
- **改动幅度验证**: 防止 LLM 过度修改原文（短文本相似度 > 30%，长文本 > 70%）
- **默认 DeepSeek，兼容所有 OpenAI 协议 API**: 无需额外配置即可使用 DeepSeek

## 安装依赖

```bash
pip install openai tenacity json-repair
```

## 快速开始

### 1. 设置 API Key

```bash
# 默认使用 DeepSeek，只需设置 key
export DEEPSEEK_API_KEY="sk-..."

# 或使用 OpenAI 兼容的其他服务
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

### 2. 基本用法

```python
from llm_subtitle_optimizer import SubtitleOptimizer

# 创建优化器（默认使用 deepseek-chat）
optimizer = SubtitleOptimizer(
    thread_num=4,           # 并发线程数
    batch_num=10,           # 每批处理的字幕条数
)

# 输入字幕字典 {index: text}
subtitles = {
    "1": "大家好啊今天呢我们来讲一下机器学习的基础只是",
    "2": "那么它其实就是嗯人工治能的一个重要份支",
    "3": "通过算发让计算机去从这个数据当中学习嘛",
}

# 优化
result = optimizer.optimize(subtitles)

for idx in sorted(result.keys(), key=int):
    print(f"  原文: {subtitles[idx]}")
    print(f"  优化: {result[idx]}")
    print()
```

### 3. 切换模型和服务商

```python
# 使用 OpenAI
optimizer = SubtitleOptimizer(
    model="gpt-4o-mini",
    base_url="https://api.openai.com/v1",
    api_key="sk-...",
)

# 使用硅基流动
optimizer = SubtitleOptimizer(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url="https://api.siliconflow.cn/v1",
    api_key="sk-...",
)
```

### 4. 直接调用 LLM

```python
from llm_subtitle_optimizer import call_llm

response = call_llm(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Say hello in Chinese."},
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)
```

## API 参考

### SubtitleOptimizer

| 参数                | 类型         | 默认值               | 说明                  |
| ----------------- | ---------- | ----------------- | ------------------- |
| `model`           | `str`      | `"deepseek-chat"` | LLM 模型名称            |
| `thread_num`      | `int`      | `4`               | 并发线程数               |
| `batch_num`       | `int`      | `10`              | 每批处理的字幕条数           |
| `custom_prompt`   | `str`      | `""`              | 自定义参考内容（术语表、上下文等）   |
| `base_url`        | `str`      | `None`            | API URL（默认 DeepSeek） |
| `api_key`         | `str`      | `None`            | API 密钥（读环境变量）       |
| `temperature`     | `float`    | `0.2`             | LLM 温度参数            |
| `update_callback` | `Callable` | `None`            | 每批完成时的回调函数          |

### 方法

- **`optimize(subtitles: Dict[str, str]) -> Dict[str, str]`**
  
  优化字幕，传入 `{"1": "text1", "2": "text2", ...}` 格式的字典，
  返回相同格式的优化后字典。

- **`optimize_from_list(texts: List[str]) -> List[str]`**
  
  从文本列表优化，是 `optimize()` 的便捷封装。

- **`shutdown()`**
  
  关闭线程池，释放资源。

## Agent Loop 工作流程

```
输入字幕 → 构建 Prompt → 调用 LLM → 解析 JSON 结果
                ↑                        ↓
                |              验证结果（键匹配 + 相似度）
                |                        ↓
                |              验证通过？—— 是 → 对齐修复 → 输出
                |                   ↓ 否
                └── 追加反馈 ←—— 生成错误描述
                                    （最多重试 3 轮）
```

## 环境变量

| 变量                  | 说明                            |
| ------------------- | ----------------------------- |
| `DEEPSEEK_API_KEY`  | DeepSeek API 密钥（优先）           |
| `OPENAI_API_KEY`    | OpenAI 兼容 API 密钥（备选）          |
| `DEEPSEEK_BASE_URL` | API 基础 URL（优先，默认 `api.deepseek.com`） |
| `OPENAI_BASE_URL`   | API 基础 URL（备选）                |

## 支持的 API 服务商

任何兼容 OpenAI API 格式的服务均可使用：

- **DeepSeek**（默认）: `https://api.deepseek.com/v1`
- **OpenAI**: `https://api.openai.com/v1`
- **硅基流动 (SiliconFlow)**: `https://api.siliconflow.cn/v1`
- **Ollama**: `http://localhost:11434/v1`
- **LM Studio**: `http://localhost:1234/v1`
- **其他兼容 OpenAI 格式的服务**

## License

Same as VideoCaptioner project.
