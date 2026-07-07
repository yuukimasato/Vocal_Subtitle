"""说话人聚类引擎

基于声学特征的凝聚聚类，将语音片段按说话人分组。

算法流程:
    1. 从完整音频中按边界提取各片段音频
    2. 计算 ~87 维声学特征向量
    3. StandardScaler 标准化 → PCA 降维 → AgglomerativeClustering
    4. 评估聚类质量（Silhouette Score）
    5. 返回片段级 speaker_id 列表

依赖: sklearn (BSD), numpy, scipy
"""

import logging
from typing import List

import numpy as np

from .base import DiarizationEngine
from .feature_extractor import FeatureExtractor

logger = logging.getLogger(__name__)


class SpeakerDiarizer(DiarizationEngine):
    """基于声学特征凝聚聚类的说话人分离器

    使用示例:
        diarizer = SpeakerDiarizer(distance_threshold=0.5)
        diarizer.load_model()  # 无实际模型加载，满足接口约定
        speaker_ids = diarizer.diarize(segments, audio, sample_rate=16000)
        # → [0, 1, 0, 0, 1, 2, 1, ...]
    """

    def __init__(
        self,
        distance_threshold: float = 0.5,
        min_speakers: int = 1,
        max_speakers: int = 10,
        use_pca: bool = True,
        pca_variance: float = 0.95,
    ):
        """
        Args:
            distance_threshold: 凝聚聚类合并阈值（余弦距离），越小越容易分成更多簇
            min_speakers: 最少说话人数（后处理约束）
            max_speakers: 最多说话人数（后处理约束）
            use_pca: 聚类前是否 PCA 降维
            pca_variance: PCA 保留的方差比例
        """
        self.distance_threshold = distance_threshold
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.use_pca = use_pca
        self.pca_variance = pca_variance

        self._extractor = FeatureExtractor()
        self._model_loaded = False
        self.last_silhouette_: float = 0.0
        self._acoustic_failed: bool = False

    # ------------------------------------------------------------------
    # DiarizationEngine 接口
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "agglomerative"

    def load_model(self) -> None:
        """延迟加载（当前纯算法实现，无模型权重）"""
        self._model_loaded = True
        logger.info("SpeakerDiarizer ready (algorithmic, no model weights)")

    @property
    def acoustic_failed(self) -> bool:
        """声学聚类是否失败（需文本降级方案）"""
        return self._acoustic_failed

    def diarize(
        self,
        segments: List,
        audio: np.ndarray,
        sample_rate: int,
    ) -> List[int]:
        """为每个语音片段分配 speaker_id

        Args:
            segments: SpeechSegment 列表（含 start/end 属性）
            audio: 完整人声音频数据 (float32, [-1, 1])
            sample_rate: 采样率

        Returns:
            与 segments 等长的 speaker_id 列表
        """
        if not segments:
            logger.warning("No segments to diarize")
            return []

        if len(segments) == 1:
            logger.info("Single segment → single speaker")
            return [0]

        # Step 1: 提取各片段的音频
        segment_audios = []
        for seg in segments:
            seg_audio = self._extract_segment_audio(seg, audio, sample_rate)
            segment_audios.append(seg_audio)

        # Step 2: 特征提取
        logger.info("Extracting features for %d segments...", len(segments))
        feature_matrix = self._extractor.extract_features_batch(
            segment_audios, sample_rate
        )

        if feature_matrix.size == 0:
            logger.warning("Feature extraction produced empty matrix")
            return [0] * len(segments)

        # Step 3: 聚类
        speaker_ids = self._cluster(feature_matrix)

        # Step 4: 质量评估
        silhouette = self._evaluate_clustering(feature_matrix, speaker_ids)
        n_speakers = len(set(speaker_ids))

        # 质量门控：检测疑似聚类失败的模式
        #   1) 低质量 (< 0.2) 且说话人数异常多 (> 3) → 可能是过度聚类
        #   2) 单说话人但段数 ≥ 5 → 多说话人对话中常见欠聚类
        #      (单簇的轮廓系数被定义为 1.0，不触发条件 1)
        needs_retry = (
            silhouette < 0
            or (silhouette < 0.20 and n_speakers > 3 and n_speakers > len(segments) // 3)
            or (n_speakers == 1 and n_segments >= 5)
        )

        if needs_retry:
            logger.warning(
                "Diarization quality is insufficient (silhouette=%.3f, %d speakers). "
                "Attempting multi-threshold retry before falling back.",
                silhouette, n_speakers,
            )
            # 尝试不同距离阈值。
            # 低阈值 (0.15-0.45) → 拆分集群获得更多说话人
            # 高阈值 (0.55+)   → 合并集群减少说话人
            is_single_cluster = (n_speakers == 1)
            if is_single_cluster:
                # 单簇情况：重点尝试低阈值以拆分出多个说话人
                retry_thresholds = [0.15, 0.20, 0.25, 0.30, 0.40]
            else:
                retry_thresholds = [0.3, 0.4, 0.6, 0.7, 0.8]

            best_labels = None
            best_score = -1.0  # 从 -1 开始，任何有效结果都能击败
            best_n_speakers = n_speakers
            original_threshold = self.distance_threshold
            try:
                for thresh in retry_thresholds:
                    self.distance_threshold = thresh
                    alt_labels = self._cluster(feature_matrix)
                    alt_score = self._evaluate_clustering(feature_matrix, alt_labels)
                    alt_n = len(set(alt_labels))
                    logger.debug(
                        "Retry threshold=%.2f → silhouette=%.3f, %d speakers",
                        thresh, alt_score, alt_n,
                    )
                    # 选择策略：
                    # - 单簇初始时：优先找 2-3 说话人且 sil > 0.1 的结果
                    # - 多簇初始时：优先找 2-3 说话人且 sil 高于当前的结果
                    if is_single_cluster:
                        if 2 <= alt_n <= 3 and alt_score > 0.1 and alt_score > best_score:
                            best_score = alt_score
                            best_labels = alt_labels
                            best_n_speakers = alt_n
                    else:
                        if alt_score > best_score and 1 <= alt_n <= 3:
                            best_score = alt_score
                            best_labels = alt_labels
                            best_n_speakers = alt_n
                        elif best_labels is None and alt_score > best_score:
                            best_score = alt_score
                            best_labels = alt_labels
                            best_n_speakers = alt_n
            finally:
                self.distance_threshold = original_threshold

            # 质量门控：仅当重试结果足够好时才采用
            quality_ok = (
                best_labels is not None
                and best_score > 0.15
                and best_n_speakers <= 3
            )
            # 单簇初始时降低门槛：sil > 0.05 且 2-3 说话人即可
            if is_single_cluster and not quality_ok:
                quality_ok = (
                    best_labels is not None
                    and best_score > 0.05
                    and 2 <= best_n_speakers <= 3
                )

            if quality_ok:
                logger.info(
                    "Multi-threshold retry succeeded: silhouette=%.3f (was %.3f), "
                    "%d speakers (was %d)",
                    best_score, silhouette, best_n_speakers, n_speakers,
                )
                speaker_ids = best_labels
                silhouette = best_score
            else:
                logger.warning(
                    "All acoustic clustering attempts failed "
                    "(best silhouette=%.3f, %d speakers). "
                    "Falling back to single speaker; "
                    "text-based diarization can re-label after ASR.",
                    best_score, best_n_speakers,
                )
                speaker_ids = [0] * len(segments)
                self._acoustic_failed = True

        logger.info(
            "Diarization complete: %d segments → %d speakers (silhouette=%.3f)",
            len(segments), len(set(speaker_ids)), silhouette,
        )

        return speaker_ids

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_segment_audio(
        seg, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """从完整音频中提取片段"""
        start_sample = max(0, int(seg.start * sample_rate))
        end_sample = min(len(audio), int(seg.end * sample_rate))
        if end_sample <= start_sample:
            return np.zeros(int(0.25 * sample_rate), dtype=np.float32)
        return audio[start_sample:end_sample].astype(np.float32)

    def _cluster(self, feature_matrix: np.ndarray) -> List[int]:
        """对特征矩阵进行凝聚聚类"""
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for speaker diarization. "
                "Install with: pip install scikit-learn"
            ) from e

        n_segments = feature_matrix.shape[0]

        # 标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(feature_matrix)

        # PCA 降维
        if self.use_pca and X.shape[1] > 3 and n_segments >= 3:
            n_components = min(n_segments - 1, X.shape[1])
            pca = PCA(n_components=self.pca_variance)
            try:
                X = pca.fit_transform(X)
                logger.debug(
                    "PCA: %d → %d components (%.1f%% variance retained)",
                    feature_matrix.shape[1], X.shape[1],
                    self.pca_variance * 100,
                )
            except Exception as e:
                logger.warning("PCA failed, using raw features: %s", e)

        # 凝聚聚类
        n_clusters = None if self.max_speakers > 1 else 1
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="cosine",
            linkage="average",
            distance_threshold=self.distance_threshold if n_clusters is None else None,
        )
        labels = clustering.fit_predict(X)

        # 后处理：约束 min/max 说话人数
        labels = self._constrain_speaker_count(
            labels, n_segments, X, self.min_speakers, self.max_speakers
        )

        return labels.tolist()

    def _constrain_speaker_count(
        self,
        labels: np.ndarray,
        n_segments: int,
        X: np.ndarray,
        min_spk: int,
        max_spk: int,
    ) -> np.ndarray:
        """后处理调整聚类数到 [min_spk, max_spk] 范围"""
        n_clusters = len(set(labels))

        if n_clusters > max_spk and max_spk >= 1:
            # 合并最近的簇
            logger.debug(
                "Merging clusters: %d → %d (max_speakers=%d)",
                n_clusters, max_spk, max_spk,
            )
            labels = self._merge_clusters(labels, X, max_spk)
        elif n_clusters < min_spk and n_segments >= min_spk:
            # 拆分最大的簇
            logger.debug(
                "Splitting clusters: %d → %d (min_speakers=%d)",
                n_clusters, min_spk, min_spk,
            )
            labels = self._split_clusters(labels, X, min_spk)

        return labels

    def _merge_clusters(
        self, labels: np.ndarray, X: np.ndarray, target: int
    ) -> np.ndarray:
        """合并最近的簇直到达到目标数"""
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            return labels

        # 计算簇中心
        unique_labels = sorted(set(labels))
        centroids = np.array([
            X[labels == lbl].mean(axis=0) for lbl in unique_labels
        ])

        # 对簇中心再次聚类
        if len(centroids) > target:
            meta = AgglomerativeClustering(
                n_clusters=target,
                metric="cosine",
                linkage="average",
            )
            meta_labels = meta.fit_predict(centroids)

            # 映射回原始标签
            new_labels = np.zeros_like(labels)
            for old_lbl, new_lbl in zip(unique_labels, meta_labels):
                new_labels[labels == old_lbl] = new_lbl
            return new_labels

        return labels

    def _split_clusters(
        self, labels: np.ndarray, X: np.ndarray, target: int
    ) -> np.ndarray:
        """拆分最大的簇直到达到目标数"""
        try:
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            return labels

        unique_labels = sorted(set(labels))
        n_current = len(unique_labels)

        # 对有多个成员的簇按大小排序
        cluster_sizes = [(lbl, (labels == lbl).sum()) for lbl in unique_labels]
        cluster_sizes.sort(key=lambda x: -x[1])  # 降序

        new_labels = labels.copy()
        next_label = max(unique_labels) + 1

        for lbl, size in cluster_sizes:
            if n_current >= target:
                break
            if size < 3:  # 太小不拆分
                continue

            # 在簇内再次聚类
            mask = new_labels == lbl
            sub_X = X[mask]
            sub_n = min(2, size - 1)
            sub = AgglomerativeClustering(
                n_clusters=sub_n,
                metric="cosine",
                linkage="average",
            )
            sub_labels = sub.fit_predict(sub_X)

            # 重新分配标签
            new_labels[mask] = np.where(
                sub_labels == 0, lbl, next_label
            )
            next_label += 1
            n_current += 1

        return new_labels

    @staticmethod
    def diarize_from_text(
        texts: List[str],
        gap_seconds: List[float],
    ) -> List[int]:
        """文本模式降级方案：用 ASR 文本推断说话人轮换。

        当声学特征聚类失败（silhouette < 0）时，分析 ASR 转录文本
        中的对话模式来推断说话人边界。

        检测的模式：
        1. 问答对（问句→陈述句）
        2. 短回应词（Yeah/OK/Got it/好的/嗯）
        3. 直接称呼（Mr./Ms./Mrs. X）
        4. 句尾标点 + 间隙 > 0.25s → 可能的说话人切换

        ★ 支持多说话人（非硬编码二元交替），使用增量式说话人分配。

        Args:
            texts: 每个段的 ASR 转录文本
            gap_seconds: 每个段与下一段的间隙（最后一个为 None 或 999）

        Returns:
            与 texts 等长的 speaker_id 列表
        """
        import re

        if len(texts) <= 1:
            return [0] * len(texts)

        n = len(texts)
        speaker_ids = [-1] * n
        next_speaker_id = 0  # 增量式分配，支持 >2 说话人

        # 模式定义
        question_pattern = re.compile(
            r'\?|吗[？?]|呢[？?]|吧[？?]|'
            r'[Hh]ow may|[Ww]hat |[Cc]an I|[Ww]hy |[Ww]hen |[Ww]here |'
            r'[Ii]s that|[Aa]re you|[Ww]ould you|[Cc]ould you|[Dd]o you|'
            r'怎么|为什么|如何|哪里|哪个|什么|谁|哪[一位]|何时|'
            r'请问|请教|怎么看|觉得|认为|能不能|可不可以|是否|'
            r'有没有|是不是|对吧|对吗|是吧|是吗|可以吗|行吗|好吗',
        )
        short_response_pattern = re.compile(
            r'^(Yeah|Yes|OK|Okay|Got it|Right|Sure|Alright|I see|'
            r'好的|好[的了]|嗯[嗯哼]|对|是的|没错|知道了|明白|My pleasure|'
            r'See you|Thank you)',
        )
        address_pattern = re.compile(
            r'\b(Mr\.|Ms\.|Mrs\.|Miss|Dr\.)\s+\w+'
            r'|'
            r'(教授|老师|先生|女士|总[监经]|博士|医生|律师|'
            r'主播|嘉宾|导演|演员|记者|解说|主持)'
        )
        sentence_end = {'.', '!', '?', '。', '！', '？'}

        # 辅助：分配一个新的 speaker ID
        def _alloc_spk() -> int:
            nonlocal next_speaker_id
            sid = next_speaker_id
            next_speaker_id += 1
            return sid

        # 辅助：获取当前已使用的最大 speaker ID + 1
        def _next_unused() -> int:
            used = {s for s in speaker_ids if s >= 0}
            if not used:
                return 0
            return max(used) + 1

        # ---- Pass 1: 问答锚点（增量式分配） ----
        for i in range(n - 1):
            if question_pattern.search(texts[i]):
                if speaker_ids[i] < 0:
                    speaker_ids[i] = _next_unused()
                if speaker_ids[i + 1] < 0:
                    # 回答者不同于提问者
                    used = {s for s in speaker_ids if s >= 0}
                    answer_spk = speaker_ids[i] + 1
                    while answer_spk in used and answer_spk == speaker_ids[i]:
                        answer_spk += 1
                    speaker_ids[i + 1] = answer_spk
                # 后续短回应保持同一说话人
                j = i + 2
                while j < n and short_response_pattern.match(texts[j].strip()):
                    if speaker_ids[j] < 0:
                        speaker_ids[j] = speaker_ids[i + 1]
                    j += 1

        # ---- Pass 2: 称呼锚点 ----
        for i in range(n):
            if address_pattern.search(texts[i]):
                if speaker_ids[i] < 0:
                    speaker_ids[i] = _next_unused()

        # ---- Pass 3: 间隙 + 短回应交替 ----
        for i in range(n - 1):
            gap = gap_seconds[i] if i < len(gap_seconds) else 999
            if gap is None or gap > 10:
                gap = 999

            if gap >= 0.25:
                if speaker_ids[i] >= 0 and speaker_ids[i + 1] < 0:
                    # 下一个说话人不同于当前
                    speaker_ids[i + 1] = _next_unused()
                elif speaker_ids[i] < 0 and speaker_ids[i + 1] >= 0:
                    speaker_ids[i] = _next_unused()
                elif speaker_ids[i] < 0 and speaker_ids[i + 1] < 0:
                    # 短回应 vs 长句 → 不同说话人
                    ti_short = bool(short_response_pattern.match(texts[i].strip()))
                    tj_short = bool(short_response_pattern.match(texts[i + 1].strip()))
                    if ti_short != tj_short:
                        speaker_ids[i] = _alloc_spk()
                        speaker_ids[i + 1] = _alloc_spk()

        # ---- Pass 4: 句尾标点 + 间隙 → 说话人切换 ----
        for i in range(n - 1):
            gap = gap_seconds[i] if i < len(gap_seconds) else 999
            if gap is None or gap > 10:
                gap = 999
            text = texts[i].rstrip()
            if text and text[-1] in sentence_end and gap > 0.25:
                if speaker_ids[i] < 0:
                    speaker_ids[i] = _next_unused()
                if speaker_ids[i + 1] < 0:
                    # 句尾+间隙 → 说话人可能切换
                    speaker_ids[i + 1] = _next_unused()

        # ---- Pass 5: 默认填充（延续上一个已知说话人） ----
        for i in range(n):
            if speaker_ids[i] < 0:
                if i > 0 and speaker_ids[i - 1] >= 0:
                    speaker_ids[i] = speaker_ids[i - 1]
                else:
                    speaker_ids[i] = 0

        # ---- 验证：确保至少有合理的说话人分布 ----
        unique = set(speaker_ids)
        if len(unique) < 2 and n > 3:
            # 所有段被归为同一说话人不合理 → 按间隙交替
            #
            # 先检查全局是否有句子结束标点（ASR 可能不输出标点）：
            has_any_sentence_end = any(
                t.rstrip() and t.rstrip()[-1] in sentence_end
                for t in texts if t.strip()
            )

            # 计算间隙中位数，用于自适应阈值
            clean_gaps = [
                g for g in gap_seconds
                if g is not None and 0 < g < 10
            ]
            if clean_gaps:
                gap_median = sorted(clean_gaps)[len(clean_gaps) // 2]
            else:
                gap_median = 0.5

            # 自适应切换阈值：取中位数的 1.5 倍，但不低于 0.15s
            switch_threshold = max(0.15, gap_median * 1.5)

            current_speaker = 0
            alt_counter = 1
            for i in range(n):
                speaker_ids[i] = current_speaker
                if i < n - 1:
                    gap = gap_seconds[i] if i < len(gap_seconds) else 999
                    if gap is None or gap > 10:
                        gap = 999
                    text = texts[i].rstrip()
                    # 有标点时使用标点 + 间隙判断（更可靠）
                    # 无标点时使用纯间隙判断（自适应阈值）
                    should_switch = (
                        gap >= 0.25 and text and text[-1] in sentence_end
                    ) or (
                        not has_any_sentence_end
                        and gap >= switch_threshold
                    )
                    if should_switch:
                        current_speaker = alt_counter
                        alt_counter = (alt_counter + 1) % max(2, 3)  # 轮换 2-3 个说话人

        # ---- 规范化 speaker ID（确保从 0 开始连续编号） ----
        unique_sorted = sorted(set(speaker_ids))
        remap = {old: new for new, old in enumerate(unique_sorted)}
        speaker_ids = [remap[s] for s in speaker_ids]

        logger.info(
            "Text-based diarization: %d segments → %d speakers",
            n, len(set(speaker_ids)),
        )
        return speaker_ids

    def _evaluate_clustering(
        self, feature_matrix: np.ndarray, labels: List[int]
    ) -> float:
        """计算聚类轮廓系数"""
        n_unique = len(set(labels))
        if n_unique < 2 or n_unique >= len(labels):
            score = 1.0 if n_unique == 1 else 0.0
            self.last_silhouette_ = score
            return score

        try:
            from sklearn.metrics import silhouette_score

            labels_arr = np.array(labels)
            # 过滤掉 -1（未分配）标签
            valid = labels_arr >= 0
            if valid.sum() < 2 or len(set(labels_arr[valid])) < 2:
                self.last_silhouette_ = 0.0
                return 0.0

            score = silhouette_score(
                feature_matrix[valid], labels_arr[valid], metric="cosine"
            )
            score = float(score)

            if score > 0.5:
                logger.info("Cluster quality: GOOD (silhouette=%.3f)", score)
            elif score > 0.25:
                logger.info("Cluster quality: ACCEPTABLE (silhouette=%.3f)", score)
            else:
                logger.warning(
                    "Cluster quality: LOW (silhouette=%.3f) — "
                    "speaker separation may be unreliable", score
                )

            self.last_silhouette_ = score
            return score
        except ImportError:
            logger.debug("sklearn not available for silhouette calculation")
            self.last_silhouette_ = 0.0
            return 0.0
        except Exception as e:
            logger.warning("Silhouette score calculation failed: %s", e)
            self.last_silhouette_ = 0.0
            return 0.0
