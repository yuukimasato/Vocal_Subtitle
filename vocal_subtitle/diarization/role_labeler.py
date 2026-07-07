"""LLM 说话人角色标注器

使用大语言模型从对话上下文中挖掘说话人名字和角色。

三级标注策略:
    1. identity → 从介绍/称呼/自述中挖掘真实名字 + 角色
    2. role     → 从对话动态中推断功能角色
    3. fallback → 无法识别时使用序列标签 "说话人A"

依赖: llm_subtitle_optimizer.llm_client.call_llm()
      llm_subtitle_optimizer.prompts.get_prompt()
"""

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 将整数 speaker_id 映射为 LLM 可读标签
_SPEAKER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _int_to_label(spk_id: int) -> str:
    """0→A, 1→B, ..., 25→Z, 26→AA, ..."""
    if spk_id < 26:
        return _SPEAKER_LABELS[spk_id]
    return _SPEAKER_LABELS[spk_id % 26] * (spk_id // 26 + 1)


class RoleLabeler:
    """LLM 说话人角色标注器

    使用示例:
        labeler = RoleLabeler()
        roles = labeler.label_roles(
            transcript_by_speaker={0: ["大家好...", "今天..."], 1: ["谢谢...", "我觉得..."]},
            model="deepseek-v4-pro",
            context_hint="a podcast interview",
        )
        # → {0: "主持人", 1: "嘉宾"}
    """

    def __init__(self):
        self._llm_client = None

    def label_roles(
        self,
        transcript_by_speaker: Dict[int, List[str]],
        model: str = "deepseek-v4-pro",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        context_hint: Optional[str] = None,
        language: str = "zh",
    ) -> Dict[int, str]:
        """为每个说话人标注角色/名字

        Args:
            transcript_by_speaker: speaker_id → 话语列表
            model: LLM 模型名称
            base_url: API URL（可选）
            api_key: API 密钥（可选）
            temperature: LLM 温度参数
            context_hint: 场景提示，如 "a podcast interview"
            language: 对话语言 "zh" | "en"

        Returns:
            speaker_id → label 映射
                e.g. {0: "张三(嘉宾)", 1: "主持人"}
        """
        if not transcript_by_speaker:
            logger.warning("Empty transcript, returning fallback labels")
            return {}

        # Step 1: 构建 LLM 输入
        conversation_text = self._build_conversation_text(transcript_by_speaker)
        system_prompt = self._get_system_prompt(context_hint, language)

        # Step 2: 调用 LLM
        try:
            response = self._call_llm(
                system_prompt=system_prompt,
                conversation=conversation_text,
                model=model,
                base_url=base_url,
                api_key=api_key,
                temperature=temperature,
            )
        except Exception as e:
            logger.error("LLM role labeling failed: %s", e)
            return self._fallback_labels(transcript_by_speaker)

        # Step 3: 解析 JSON 响应
        parsed = self._parse_response(response, transcript_by_speaker)
        if parsed is None:
            return self._fallback_labels(transcript_by_speaker)

        # Step 4: 转换标签: "A"→0, "B"→1, ... →提取 label 字段
        return self._convert_labels(parsed, transcript_by_speaker)

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_conversation_text(
        self, transcript_by_speaker: Dict[int, List[str]]
    ) -> str:
        """构建格式化的对话文本"""
        parts = []
        for spk_id in sorted(transcript_by_speaker.keys()):
            label = f"说话人{_int_to_label(spk_id)}"
            utterances = transcript_by_speaker[spk_id]

            # 限制每个说话人的话语长度（避免 token 溢出）
            truncated = self._truncate_utterances(utterances)

            parts.append(f"## {label}")
            for utt in truncated:
                parts.append(f"  {utt}")
            parts.append("")

        return "\n".join(parts).strip()

    @staticmethod
    def _truncate_utterances(
        utterances: List[str], max_chars: int = 3000
    ) -> List[str]:
        """截断过长的话语序列，保留开头和结尾"""
        total = 0
        kept = []
        for utt in utterances:
            total += len(utt)
            kept.append(utt)
            if total > max_chars:
                break

        # 如果截断了，从末尾追加一些
        if len(kept) < len(utterances):
            tail = []
            tail_chars = 0
            for utt in reversed(utterances):
                tail_chars += len(utt)
                if tail_chars > 1000:
                    break
                tail.insert(0, utt)
            # 避免重复
            existing = set(kept)
            for utt in tail:
                if utt not in existing:
                    kept.append(utt)

        return kept

    def _get_system_prompt(
        self, context_hint: Optional[str], language: str
    ) -> str:
        """加载并填充 prompt 模板"""
        try:
            from llm_subtitle_optimizer.prompts import get_prompt
        except ImportError:
            raise ImportError(
                "llm_subtitle_optimizer is required for role labeling"
            )

        hint = context_hint or (
            "a multi-speaker conversation" if language == "zh"
            else "a multi-speaker conversation"
        )
        return get_prompt(
            "speaker_role",
            context_hint=hint,
            language=language,
        )

    def _call_llm(
        self,
        system_prompt: str,
        conversation: str,
        model: str,
        base_url: Optional[str],
        api_key: Optional[str],
        temperature: float,
    ) -> str:
        """调用 LLM API"""
        try:
            from llm_subtitle_optimizer.llm_client import call_llm
        except ImportError:
            raise ImportError(
                "llm_subtitle_optimizer is required for role labeling"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": conversation},
        ]

        response = call_llm(
            messages=messages,
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
        )

        content = response.choices[0].message.content
        if not content:
            raise ValueError("LLM returned empty response")
        return content

    def _parse_response(
        self, response_text: str, transcript_by_speaker: Dict[int, List[str]]
    ) -> Optional[Dict[str, Any]]:
        """解析 LLM JSON 响应，失败则重试一次"""

        def _try_parse(text: str) -> Optional[Dict[str, Any]]:
            # 去除可能的 markdown 代码块
            text = text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
                text = text.strip()

            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass

            # 尝试用 json-repair 修复
            try:
                from json_repair import repair_json
                fixed = repair_json(text)
                return json.loads(fixed)
            except Exception:
                pass

            return None

        result = _try_parse(response_text)
        if result is not None and self._validate_keys(result, transcript_by_speaker):
            return result

        # 重试：追加 JSON 格式要求
        logger.warning("First parse attempt failed, retrying with stricter prompt...")
        try:
            retry_content = self._call_llm(
                system_prompt=(
                    f"{response_text}\n\n"
                    "The above response was not valid JSON. "
                    "Please output ONLY a valid JSON object with these exact keys: "
                    f"{json.dumps([_int_to_label(i) for i in sorted(transcript_by_speaker.keys())])}\n"
                    "Output pure JSON without any markdown or explanation."
                ),
                conversation="Please provide the speaker role JSON for the conversation you just analyzed.",
                model="deepseek-v4-pro",  # 重试用默认模型
                base_url=None,
                api_key=None,
                temperature=0.1,  # 更低温度确保格式正确
            )
            result = _try_parse(retry_content)
        except Exception as e:
            logger.warning("Retry also failed: %s", e)
            result = None

        if result is not None and self._validate_keys(result, transcript_by_speaker):
            return result

        logger.warning("Failed to parse LLM response after retry")
        return None

    @staticmethod
    def _validate_keys(
        parsed: Dict[str, Any], transcript_by_speaker: Dict[int, List[str]]
    ) -> bool:
        """验证 JSON 是否包含所有说话人的键"""
        expected_keys = {_int_to_label(i) for i in transcript_by_speaker.keys()}
        actual_keys = set(parsed.keys())
        if not expected_keys.issubset(actual_keys):
            missing = expected_keys - actual_keys
            logger.warning("LLM response missing keys: %s", missing)
            return False
        return True

    def _convert_labels(
        self,
        parsed: Dict[str, Any],
        transcript_by_speaker: Dict[int, List[str]],
    ) -> Dict[int, str]:
        """将 LLM 响应转换为 speaker_id → label 映射

        处理两种 JSON 格式:
        - 完整格式: {"A": {"name": "...", "role": "...", "label": "...", "confidence": "..."}}
        - 简化格式: {"A": "角色名"}
        """
        result = {}
        for spk_id in sorted(transcript_by_speaker.keys()):
            key = _int_to_label(spk_id)
            value = parsed.get(key)

            if isinstance(value, dict):
                label = value.get("label", "")
                if not label:
                    # 从 name + role 构建
                    name = value.get("name", "")
                    role = value.get("role", "")
                    if name and role:
                        label = f"{name}({role})"
                    elif name:
                        label = name
                    elif role:
                        label = role
                    else:
                        label = self._generic_label(spk_id)
                result[spk_id] = label
            elif isinstance(value, str):
                result[spk_id] = value
            else:
                result[spk_id] = self._generic_label(spk_id)

        return result

    def _fallback_labels(
        self, transcript_by_speaker: Dict[int, List[str]]
    ) -> Dict[int, str]:
        """LLM 不可用时的兜底标签"""
        return {
            spk_id: self._generic_label(spk_id)
            for spk_id in transcript_by_speaker.keys()
        }

    @staticmethod
    def _generic_label(spk_id: int) -> str:
        """生成通用说话人标签"""
        return f"说话人{_int_to_label(spk_id)}"
