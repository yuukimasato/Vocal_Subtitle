"""LLM 说话人角色标注器测试

使用 mock LLM 响应验证三级标注策略：
    identity → role → fallback
"""

import json
from unittest import mock

import pytest

from vocal_subtitle.diarization.role_labeler import RoleLabeler, _int_to_label


# ---------------------------------------------------------------------------
# Mock 工具
# ---------------------------------------------------------------------------


class _FakeResponse:
    """模拟 openai API 响应对象（含 choices[0].message.content）"""

    def __init__(self, content: str):
        self.choices = [_FakeChoice(content)]


class _FakeChoice:
    """模拟 openai.types.chat.ChatCompletionChoice"""

    def __init__(self, content: str):
        self.message = _FakeMessage(content)


class _FakeMessage:
    def __init__(self, content: str):
        self.content = content


def _mock_identity_response(speaker_ids: list[int]) -> str:
    """生成 identity 级别的 mock JSON 响应。

    模拟场景通讯: 说话人0 是主持人（提问、介绍嘉宾），说话人1 是嘉宾（被介绍、回答）。
    LLM 分析后推断: A=主持人, B=李教授(嘉宾)
    """
    data = {}
    for spk_id in speaker_ids:
        key = _int_to_label(spk_id)
        if spk_id == 0:
            data[key] = {
                "name": None,
                "role": "主持人",
                "label": "主持人",
                "confidence": "role",
            }
        elif spk_id == 1:
            data[key] = {
                "name": "李教授",
                "role": "嘉宾",
                "label": "李教授(嘉宾)",
                "confidence": "identity",
            }
        else:
            data[key] = {
                "name": None,
                "role": None,
                "label": f"说话人{key}",
                "confidence": "fallback",
            }
    return json.dumps(data)


def _mock_role_only_response(speaker_ids: list[int]) -> str:
    """生成 role-only 级别的 mock JSON 响应。

    模拟场景: 说话人0是主持人（开场、介绍话题、邀请嘉宾），说话人1是嘉宾（被邀请、表达观点、感谢主持人）。
    LLM 推断: A=主持人, B=嘉宾
    """
    data = {}
    for spk_id in speaker_ids:
        key = _int_to_label(spk_id)
        if spk_id == 0:
            data[key] = {"name": None, "role": "主持人", "label": "主持人", "confidence": "role"}
        elif spk_id == 1:
            data[key] = {"name": None, "role": "嘉宾", "label": "嘉宾", "confidence": "role"}
        else:
            data[key] = {"name": None, "role": None, "label": f"说话人{key}", "confidence": "fallback"}
    return json.dumps(data)


def _make_mock_call_llm(json_content: str):
    """创建 mock 的 call_llm 函数，返回指定 JSON 内容"""

    def _caller(
        messages=None,
        model=None,
        temperature=None,
        base_url=None,
        api_key=None,
        client=None,
        **kwargs,
    ):
        return _FakeResponse(json_content)

    return _caller


# ---------------------------------------------------------------------------
# 测试类
# ---------------------------------------------------------------------------


class TestIntToLabel:
    """说话人 ID 到标签字母映射"""

    def test_zero_is_a(self):
        assert _int_to_label(0) == "A"

    def test_one_is_b(self):
        assert _int_to_label(1) == "B"

    def test_twenty_five_is_z(self):
        assert _int_to_label(25) == "Z"

    def test_twenty_six_is_aa(self):
        assert _int_to_label(26) == "AA"


class TestRoleLabelerFallback:
    """回落行为测试（LLM 不可用/无网络环境）"""

    def test_empty_transcript_returns_empty(self):
        """空转录文本 → 空字典"""
        labeler = RoleLabeler()
        result = labeler.label_roles({})
        assert result == {}

    def test_llm_error_falls_back(self):
        """LLM 调用失败 → 回落至通用标签（不崩溃）"""
        labeler = RoleLabeler()
        transcript = {
            0: ["大家好，欢迎收听今天的节目。"],
            1: ["谢谢主持人，很高兴来到这里。"],
        }

        # 模拟 LLM 抛出异常（call_llm 在 _call_llm 方法中延迟导入）
        with mock.patch(
            "llm_subtitle_optimizer.llm_client.call_llm",
            side_effect=RuntimeError("API unavailable"),
        ):
            result = labeler.label_roles(transcript)

        assert isinstance(result, dict)
        assert len(result) == 2
        # 应使用通用标签
        assert 0 in result
        assert 1 in result
        assert "说话人" in result[0]
        assert "说话人" in result[1]

    def test_invalid_json_falls_back(self):
        """LLM 返回无效 JSON → 回落至通用标签"""
        labeler = RoleLabeler()
        transcript = {
            0: ["Hello."],
            1: ["Hi."],
        }

        # 第一次返回无效 JSON，重试也返回无效 JSON
        call_count = [0]

        def _bad_json(**kwargs):
            call_count[0] += 1
            # 返回一个 json_repair 也无法修复为 dict 的字符串
            return _FakeResponse("definitely not json at all 12345")

        with mock.patch(
            "llm_subtitle_optimizer.llm_client.call_llm",
            side_effect=_bad_json,
        ):
            result = labeler.label_roles(transcript)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert result[0] == "说话人A"
        assert result[1] == "说话人B"


class TestRoleLabelerIdentity:
    """identity 级别：姓名挖掘"""

    def test_identity_level_with_names(self):
        """LLM 返回姓名+角色 → 合并标签 '姓名(角色)'"""
        labeler = RoleLabeler()
        transcript = {
            0: [
                "今天我们请到了李教授来做客。",
                "李教授，您对这个话题有什么看法？",
            ],
            1: [
                "谢谢主持人，很高兴来到这里。",
                "我觉得这个问题可以从几个角度来看。",
            ],
        }

        mock_llm = _make_mock_call_llm(_mock_identity_response([0, 1]))

        with mock.patch(
            "llm_subtitle_optimizer.llm_client.call_llm",
            side_effect=mock_llm,
        ):
            result = labeler.label_roles(transcript)

        assert result[1] == "李教授(嘉宾)"
        assert result[0] == "主持人"

    def test_identity_json_missing_key_falls_back(self):
        """JSON 缺少某个说话人的键 → 回落"""
        labeler = RoleLabeler()
        transcript = {0: ["Hello."], 1: ["Hi."], 2: ["Hey."]}

        # 只返回 A, B，缺少 C
        incomplete_json = json.dumps({
            "A": {"name": "Alice", "role": "Host", "label": "Alice(Host)", "confidence": "identity"},
            "B": {"name": "Bob", "role": "Guest", "label": "Bob(Guest)", "confidence": "identity"},
            # 缺少 C
        })

        mock_llm = _make_mock_call_llm(incomplete_json)

        with mock.patch(
            "llm_subtitle_optimizer.llm_client.call_llm",
            side_effect=mock_llm,
        ):
            result = labeler.label_roles(transcript)

        # 验证失败 → 全部回落
        for spk_id in transcript:
            assert "说话人" in result[spk_id]


class TestRoleLabelerRoleOnly:
    """role-only 级别：角色推断（无姓名）"""

    def test_role_inference_without_names(self):
        """仅有角色信息 → 直接使用角色名称"""
        labeler = RoleLabeler()
        transcript = {
            0: [
                "大家好，欢迎收看今天的节目。",
                "今天我们讨论的话题是人工智能。",
                "下面请我们的嘉宾来做个自我介绍。",
            ],
            1: [
                "大家好，很高兴来到这里。",
                "我认为AI将会改变世界。",
                "谢谢主持人的邀请。",
            ],
        }

        mock_llm = _make_mock_call_llm(_mock_role_only_response([0, 1]))

        with mock.patch(
            "llm_subtitle_optimizer.llm_client.call_llm",
            side_effect=mock_llm,
        ):
            result = labeler.label_roles(transcript)

        assert result[0] == "主持人"
        assert result[1] == "嘉宾"

    def test_simplified_string_format(self):
        """JSON 值为简单字符串格式（非完整对象）"""
        labeler = RoleLabeler()
        transcript = {0: ["Hello."], 1: ["Hi."]}

        simple_json = json.dumps({"A": "主持", "B": "嘉宾"})

        mock_llm = _make_mock_call_llm(simple_json)

        with mock.patch(
            "llm_subtitle_optimizer.llm_client.call_llm",
            side_effect=mock_llm,
        ):
            result = labeler.label_roles(transcript)

        assert result[0] == "主持"
        assert result[1] == "嘉宾"


class TestTruncation:
    """话语截断测试"""

    def test_truncate_long_transcript(self):
        """过长话语被截断以避免 token 溢出"""
        labeler = RoleLabeler()
        long_text = "测试文本。" * 500  # ~2500 字符
        utterances = [long_text] * 10

        truncated = labeler._truncate_utterances(utterances, max_chars=3000)
        # 截断后总字符数应 ≤ max_chars + 尾部追加（~1000）
        total = sum(len(u) for u in truncated)
        assert total <= 5000, f"Truncated total {total} exceeds expected limit"
        assert len(truncated) < len(utterances), "Should have truncated some utterances"


class TestBuildConversationText:
    """对话文本构建测试"""

    def test_formats_speakers_correctly(self):
        """对话文本按说话人正确分组"""
        labeler = RoleLabeler()
        transcript = {
            0: ["你好。", "今天天气不错。"],
            1: ["确实如此。", "我们出去走走吧。"],
        }

        text = labeler._build_conversation_text(transcript)

        assert "说话人A" in text
        assert "说话人B" in text
        assert "你好。" in text
        assert "确实如此。" in text
