"""字幕优化模块

使用 LLM 优化字幕内容，支持 agent loop 自动验证和修正。

核心功能:
- Agent loop: LLM → 验证 → 反馈 → 重试（最多3轮）
- 并发批量处理
- 自动对齐修复（处理优化导致的段落合并/拆分）
- 改动幅度验证（防止 LLM 过度修改）
"""

import difflib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple, Union

from .aligner import SubtitleAligner
from .llm_client import call_llm
from .prompts import get_prompt
from .text_utils import count_words

MAX_STEPS = 3


def _ensure_json_repair():
    """按需加载 json_repair（LLM extra 的一部分）。"""
    try:
        import json_repair as _jr
        return _jr
    except ImportError:
        raise ImportError(
            "LLM 功能需要 json-repair 库，请安装: pip install -e '.[llm]'\n"
            "或直接: pip install json-repair"
        )


class SubtitleOptimizer:
    """字幕优化器

    使用 LLM 优化字幕内容，支持:
    - Agent loop 自动验证和修正
    - 并发批量处理
    - 自动对齐修复

    Usage:
        optimizer = SubtitleOptimizer(
            model="deepseek-v4-pro",
            thread_num=4,
            batch_num=10,
        )

        # 输入: {index: text} 字典
        subtitles = {"1": "大家好啊今天呢...", "2": "那么它其实就是...", ...}

        # 优化
        result = optimizer.optimize(subtitles)
        # result: {"1": "大家好啊，今天...", "2": "那么它其实就是...", ...}
    """

    def __init__(
        self,
        model: str = "deepseek-v4-pro",
        thread_num: int = 4,
        batch_num: int = 10,
        custom_prompt: str = "",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        update_callback: Optional[Callable] = None,
    ):
        """初始化优化器

        Args:
            model: LLM 模型名称
            thread_num: 并发线程数
            batch_num: 每批处理的字幕数量
            custom_prompt: 自定义优化参考内容（如术语表、上下文等）
            base_url: API 基础 URL（默认读取环境变量）
            api_key: API 密钥（默认读取环境变量）
            temperature: LLM 温度参数（默认 0.2，低温度保证一致性）
            update_callback: 进度回调函数，
                签名为 callback(batch_result: list[dict])，
                每批完成后调用
        """
        self.model = model
        self.thread_num = thread_num
        self.batch_num = batch_num
        self.custom_prompt = custom_prompt
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.update_callback = update_callback

        self._executor: Optional[ThreadPoolExecutor] = None

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """延迟创建线程池"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self.thread_num)
        return self._executor

    def optimize(
        self, subtitles: Dict[str, str],
        event_metadata: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, str]:
        """优化字幕

        Args:
            subtitles: 字幕字典 {index: text}，index 为字符串数字如 "1"
            event_metadata: 可选的字幕元数据 {index: {start, end, speaker, ...}}

        Returns:
            优化后的字幕字典 {index: optimized_text}
        """
        if not subtitles:
            return {}

        # 分批处理（文本 + 元数据一起分块）
        chunks = self._split_chunks(subtitles, event_metadata)

        # 并行优化
        optimized_dict = self._parallel_optimize(chunks)

        return optimized_dict

    def optimize_from_list(
        self, texts: List[str]
    ) -> List[str]:
        """从文本列表优化字幕（便捷方法）

        Args:
            texts: 字幕文本列表，按顺序排列

        Returns:
            优化后的文本列表
        """
        subtitle_dict = {str(i + 1): text for i, text in enumerate(texts)}
        result = self.optimize(subtitle_dict)
        # 按原始顺序返回
        return [
            result.get(str(i + 1), texts[i])
            for i in range(len(texts))
        ]

    def _split_chunks(
        self, subtitle_dict: Dict[str, str],
        event_metadata: Optional[Dict[str, Dict]] = None,
    ) -> List[Tuple[Dict[str, str], Optional[Dict[str, Dict]]]]:
        """将字幕字典分割成批次，同时分割对应的元数据"""
        items = list(subtitle_dict.items())
        chunks = []
        for i in range(0, len(items), self.batch_num):
            chunk_items = items[i : i + self.batch_num]
            chunk_dict = dict(chunk_items)
            # 分割对应的元数据
            chunk_meta = None
            if event_metadata:
                chunk_meta = {
                    idx: event_metadata[idx]
                    for idx in chunk_dict
                    if idx in event_metadata
                }
                if not chunk_meta:
                    chunk_meta = None
            chunks.append((chunk_dict, chunk_meta))
        return chunks

    def _parallel_optimize(
        self, chunks: List[Tuple[Dict[str, str], Optional[Dict[str, Dict]]]]
    ) -> Dict[str, str]:
        """并行优化所有批次"""
        executor = self._ensure_executor()
        optimized_dict: Dict[str, str] = {}

        futures = {
            executor.submit(self._optimize_chunk, chunk, meta): chunk
            for chunk, meta in chunks
        }

        for future in as_completed(futures):
            chunk = futures[future]
            try:
                result = future.result()
                optimized_dict.update(result)
            except Exception as e:
                # 失败时保留原文
                optimized_dict.update(chunk)

        return optimized_dict

    def _optimize_chunk(
        self, subtitle_chunk: Dict[str, str],
        event_metadata: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, str]:
        """优化单个字幕批次"""
        start_idx = next(iter(subtitle_chunk))
        end_idx = next(reversed(subtitle_chunk))

        try:
            result = self.agent_loop(subtitle_chunk, event_metadata)

            # 调用进度回调
            if self.update_callback:
                callback_data = [
                    {
                        "index": int(idx),
                        "original_text": subtitle_chunk[idx],
                        "optimized_text": result[idx],
                    }
                    for idx in sorted(result.keys(), key=int)
                ]
                self.update_callback(callback_data)

            return result

        except Exception as e:
            return subtitle_chunk

    def agent_loop(
        self, subtitle_chunk: Dict[str, str],
        event_metadata: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, str]:
        """使用 agent loop 优化字幕

        LLM → 验证 → 反馈 → 重试 (最多 MAX_STEPS 次)

        Args:
            subtitle_chunk: 字幕批次字典
            event_metadata: 可选的字幕元数据 {index: {start, end, speaker, ...}}

        Returns:
            优化后的字幕批次

        Raises:
            ValueError: LLM 返回空结果
        """
        # 构建提示词 — 如果有元数据则附带上下文，否则退化为纯文本
        if event_metadata:
            # 构建带上下文的输入
            context_entries = {}
            for idx in sorted(subtitle_chunk.keys(), key=int):
                text = subtitle_chunk[idx]
                meta = event_metadata.get(idx, {})
                entry = {"text": text}
                if meta.get("speaker"):
                    entry["speaker"] = meta["speaker"]
                if meta.get("start") is not None and meta.get("end") is not None:
                    entry["time"] = f"{meta['start']:.1f}s - {meta['end']:.1f}s"
                # 相邻条目信息
                if meta.get("next_speaker") and meta["next_speaker"] != meta.get("speaker"):
                    entry["next_speaker"] = f"{meta['next_speaker']} (DIFFERENT)"
                if meta.get("prev_speaker") and meta["prev_speaker"] != meta.get("speaker"):
                    entry["prev_speaker"] = f"{meta['prev_speaker']} (DIFFERENT)"
                context_entries[idx] = entry
            input_json = json.dumps(context_entries, ensure_ascii=False, indent=2)
        else:
            input_json = str(subtitle_chunk)

        user_prompt = (
            "Correct the following subtitles. "
            "Keep the original language, do not translate.\n\n"
            "MANDATORY RULE — VIOLATION WILL CAUSE OUTPUT TO BE REJECTED:\n"
            "Each entry has a FIXED time position. You MUST NOT move,\n"
            "copy, or relocate ANY text from one entry to another.\n"
            "Even if two entries are semantically related, their text\n"
            "must stay SEPARATE. Fix only spelling, grammar, and punctuation\n"
            "WITHIN each individual entry.\n\n"
            f"<input_subtitle>{input_json}</input_subtitle>"
        )

        if self.custom_prompt:
            user_prompt += (
                f"\nReference content:\n"
                f"<reference>{self.custom_prompt}</reference>"
            )

        messages = [
            {"role": "system", "content": get_prompt("subtitle")},
            {"role": "user", "content": user_prompt},
        ]

        last_result = None

        # Agent loop
        for step in range(MAX_STEPS):
            # 调用 LLM
            response = call_llm(
                messages=messages,
                model=self.model,
                temperature=self.temperature,
                base_url=self.base_url,
                api_key=self.api_key,
            )

            result_text = response.choices[0].message.content
            if not result_text:
                raise ValueError("LLM returned empty result")

            # 解析结果
            parsed_result = _ensure_json_repair().loads(result_text)
            if not isinstance(parsed_result, dict):
                raise ValueError(
                    f"LLM returned unexpected type, "
                    f"expected dict, got {type(parsed_result)}"
                )

            # 规范化输出：处理 LLM 可能返回嵌套对象（匹配输入格式）的情况
            result_dict: Dict[str, str] = {}
            for k, v in parsed_result.items():
                if isinstance(v, dict):
                    # LLM 返回了嵌套对象 → 提取 text 字段
                    result_dict[str(k)] = str(v.get("text", v))
                elif isinstance(v, str):
                    result_dict[str(k)] = v
                else:
                    result_dict[str(k)] = str(v)
            last_result = result_dict

            # 验证结果
            is_valid, error_message = self._validate(
                original_chunk=subtitle_chunk,
                optimized_chunk=result_dict,
            )

            if is_valid:
                # ★ 先清除吸收条目，再对齐修复（避免 aligner 干扰吸收检测）
                absorbed_cleaned = self._resolve_absorbed_entries(
                    subtitle_chunk, result_dict, event_metadata,
                )
                repaired = self._repair(subtitle_chunk, absorbed_cleaned)
                return repaired

            # 验证失败，添加反馈
            messages.append(
                {"role": "assistant", "content": result_text}
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Validation failed: {error_message}\n"
                        f"Please fix the errors and output "
                        f"ONLY a valid JSON dictionary."
                    ),
                }
            )

        # 达到最大步数，返回最后一次结果
        if last_result:
            absorbed_cleaned = self._resolve_absorbed_entries(
                subtitle_chunk, last_result, event_metadata,
            )
            repaired = self._repair(subtitle_chunk, absorbed_cleaned)
            return repaired
        return subtitle_chunk

    def _validate(
        self,
        original_chunk: Dict[str, str],
        optimized_chunk: Dict[str, str],
    ) -> Tuple[bool, str]:
        """验证优化结果

        检查:
        1. 键是否完全匹配
        2. 改动是否过大（相似度阈值检测）

        Args:
            original_chunk: 原始字幕批次
            optimized_chunk: 优化后字幕批次

        Returns:
            (是否有效, 错误反馈)
        """
        expected_keys = set(original_chunk.keys())
        actual_keys = set(optimized_chunk.keys())

        # 检查键匹配
        if expected_keys != actual_keys:
            missing = expected_keys - actual_keys
            extra = actual_keys - expected_keys

            error_parts = []
            if missing:
                error_parts.append(f"Missing keys: {sorted(missing)}")
            if extra:
                error_parts.append(f"Extra keys: {sorted(extra)}")

            error_msg = (
                "\n".join(error_parts)
                + f"\nRequired keys: {sorted(expected_keys)}\n"
                f"Please return the COMPLETE optimized dictionary "
                f"with ALL {len(expected_keys)} keys."
            )
            return False, error_msg

        # 检查改动是否过大（逐条比较相似度）
        excessive_changes = []
        for key in expected_keys:
            original_text = original_chunk[key]
            optimized_text = optimized_chunk[key]

            # ★ 类型安全检查：确保优化后的值是字符串
            if not isinstance(optimized_text, str):
                return False, (
                    f"Key '{key}': optimized value is not a string "
                    f"(got {type(optimized_text).__name__}). "
                    f"Return ONLY a flat JSON object with string values."
                )
            if not isinstance(original_text, str):
                original_text = str(original_text)

            # 清理文本用于比较
            original_cleaned = re.sub(r"\s+", " ", original_text).strip()
            optimized_cleaned = re.sub(r"\s+", " ", optimized_text).strip()

            # 计算相似度
            matcher = difflib.SequenceMatcher(
                None, original_cleaned, optimized_cleaned
            )
            similarity = matcher.ratio()
            similarity_threshold = (
                0.6 if count_words(original_text) <= 10 else 0.7
            )

            # 相似度过低
            if similarity < similarity_threshold:
                excessive_changes.append(
                    f"Key '{key}': "
                    f"similarity {similarity:.1%} < {similarity_threshold:.0%}. "
                    f"Original: '{original_text}' → "
                    f"Optimized: '{optimized_text}' "
                )

        if excessive_changes:
            error_msg = ";\n".join(excessive_changes)
            error_msg += (
                "\n\nYour optimizations changed the text too much. "
                "Keep high similarity (≥70% for normal text) "
                "by making MINIMAL changes: "
                "only fix recognition errors and improve clarity, "
                "but preserve the original wording, length and structure "
                "as much as possible."
            )
            return False, error_msg

        # ★ 跨键去重检查：检测文本是否从一个条目复制到另一个
        sorted_keys = sorted(expected_keys, key=int)
        for i, key_a in enumerate(sorted_keys):
            text_a = optimized_chunk[key_a].strip()
            if len(text_a) < 3:
                continue
            for key_b in sorted_keys[i + 1:]:
                text_b = optimized_chunk[key_b].strip()
                if len(text_b) < 3:
                    continue
                # 检测完全相同
                if text_a == text_b:
                    return False, (
                        f"DUPLICATE TEXT DETECTED: Key '{key_a}' and '{key_b}' "
                        f"contain identical text ('{text_a[:50]}'). "
                        f"Remove the duplication. Each subtitle entry must contain "
                        f"only its own unique content."
                    )
                # 检测高度相似（>90%）
                if len(text_a) > 5 and len(text_b) > 5:
                    similarity = difflib.SequenceMatcher(None, text_a, text_b).ratio()
                    if similarity > 0.9:
                        return False, (
                            f"DUPLICATE TEXT DETECTED: Key '{key_a}' and '{key_b}' "
                            f"contain near-identical text (similarity {similarity:.0%}). "
                            f"Remove the duplication. Each subtitle entry must contain "
                            f"only its own unique content."
                        )
                # ★ 检测子串包含：较短文本被完全包含在较长文本中
                shorter = text_a if len(text_a) <= len(text_b) else text_b
                longer = text_b if len(text_a) <= len(text_b) else text_a
                if len(shorter) >= 4 and shorter in longer:
                    short_key = key_a if len(text_a) <= len(text_b) else key_b
                    long_key = key_b if len(text_a) <= len(text_b) else key_a
                    return False, (
                        f"TEXT ABSORPTION DETECTED: Key '{long_key}' contains the "
                        f"entire text of key '{short_key}' ('{shorter}'). "
                        f"Each subtitle entry must contain only its own content. "
                        f"Do not copy or move text between entries."
                    )

        return True, ""

    @staticmethod
    def _resolve_absorbed_entries(
        original: Dict[str, str],
        optimized: Dict[str, str],
        event_metadata: Optional[Dict[str, Dict]] = None,
    ) -> Dict[str, str]:
        """清除被上一句吸收了内容的冗余条目。

        当 LLM 将下一句的内容追加到当前句末尾时（如将"四个半."
        优化为"四个半. 你中间歇了八回."），下一句变为冗余，应清空
        以便上游过滤移除。

        检测规则：
        1. 如果 optimized[N] 完整包含了 original[N+1] 或 optimized[N+1]
           的文本内容，则清空 optimized[N+1]，并从 optimized[N] 中移除该内容。
        2. 支持模糊匹配：LLM 可能对吸收的文本做了微调（加标点等），
           使用清理后的文本做归一化比较。
        3. 不同说话人之间不执行吸收检测（硬边界）。
        4. 检查窗口内的所有后续条目，不因中间条目未吸收而提前停止。

        Args:
            original: 原始字幕字典
            optimized: 优化后字幕字典
            event_metadata: 可选的字幕元数据，用于说话人边界检查

        Returns:
            清除冗余条目后的字幕字典
        """
        if len(optimized) <= 1:
            return optimized

        sorted_keys = sorted(optimized.keys(), key=int)

        for i in range(len(sorted_keys) - 1):
            curr_key = sorted_keys[i]
            curr_text = optimized[curr_key].strip()
            if not curr_text:
                continue

            # 检查后续条目（窗口内全部检查，不提前 break）
            for j in range(i + 1, len(sorted_keys)):
                next_key = sorted_keys[j]
                next_opt = optimized[next_key].strip()
                if not next_opt:
                    continue

                # ★ 说话人硬边界：不同说话人之间绝不吸收
                if event_metadata:
                    curr_meta = event_metadata.get(curr_key, {})
                    next_meta = event_metadata.get(next_key, {})
                    curr_spk = curr_meta.get("speaker")
                    next_spk = next_meta.get("speaker")
                    if curr_spk and next_spk and curr_spk != next_spk:
                        continue  # 不同说话人 → 跳过此条目，继续检查后续

                next_orig = original.get(next_key, "").strip()

                # 取较短者作为被吸收内容的参照
                absorbed_text = (
                    next_orig if len(next_orig) <= len(next_opt) else next_opt
                )

                # ---- 精确匹配 ----
                if len(absorbed_text) >= 3 and absorbed_text in curr_text:
                    optimized[next_key] = ""
                    # ★ 同时从吸收者中移除被吸收的文本
                    optimized[curr_key] = curr_text.replace(
                        absorbed_text, ""
                    ).strip().rstrip(".,，。;；").strip()
                    curr_text = optimized[curr_key]  # 更新引用，用于后续检测
                    continue

                # ---- 模糊匹配：清理标点后比较 ----
                cleaned_absorbed = _clean_for_compare(absorbed_text)
                cleaned_curr = _clean_for_compare(curr_text)
                if len(cleaned_absorbed) >= 3 and cleaned_absorbed in cleaned_curr:
                    # 在原文中找到对应位置并移除
                    idx = cleaned_curr.find(cleaned_absorbed)
                    if idx >= 0:
                        optimized[next_key] = ""
                        optimized[curr_key] = (
                            curr_text[:idx] + curr_text[idx + len(cleaned_absorbed):]
                        ).strip().rstrip(".,，。;；").strip()
                        curr_text = optimized[curr_key]
                    continue

        return optimized


def _clean_for_compare(text: str) -> str:
    """清理文本用于吸收比较：去除标点差异和多余空格"""
    import re as _re
    t = text.strip()
    # 移除末尾标点（LLM 常添加的）
    t = _re.sub(r'[.。！!？?，,；;、]+$', '', t)
    # 移除开头标点
    t = _re.sub(r'^[.。！!？?，,；;、]+', '', t)
    # 规范化空白
    t = _re.sub(r'\s+', '', t)
    return t

    @staticmethod
    def _repair(
        original: Dict[str, str],
        optimized: Dict[str, str],
    ) -> Dict[str, str]:
        """修复字幕对齐

        使用 SubtitleAligner 对齐原文和优化后的文本，
        处理优化过程中可能产生的段落合并或拆分。

        Args:
            original: 原始字幕字典
            optimized: 优化后字幕字典

        Returns:
            对齐后的字幕字典
        """
        try:
            aligner = SubtitleAligner()
            original_list = list(original.values())
            optimized_list = list(optimized.values())

            aligned_source, aligned_target = aligner.align_texts(
                original_list, optimized_list
            )

            if len(aligned_source) != len(aligned_target):
                return optimized

            # 重建字典，保持原有索引
            start_id = next(iter(original.keys()))
            return {
                str(int(start_id) + i): text
                for i, text in enumerate(aligned_target)
            }

        except Exception:
            return optimized

    def shutdown(self):
        """关闭线程池"""
        if self._executor:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
