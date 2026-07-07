"""音频声学指纹提取与匹配 (Phase 5.3)

提取 48 维声学特征向量，用于相似音频自动匹配。
匹配策略：马氏距离 + KNN 动态阈值 + Z-score 标准化。

数据存储：SQLite fingerprints.db
"""

import hashlib
import json
import logging
import sqlite3
import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class AudioFingerprint:
    """音频声学特征向量（用于相似音频匹配）

    共 48 维特征:
        频谱特征 (1+1+7+13 = 22维):
            spectral_centroid_mean, spectral_bandwidth_mean,
            spectral_contrast_mean (7维), mfcc_means (13维)
        时域特征 (3维):
            rms_mean, rms_std, zero_crossing_rate
        语音特征 (2维):
            speech_ratio, estimated_speaker_count
        噪声特征 (2维):
            noise_floor_db, snr_estimate
        基础信息 (2维):
            duration_seconds, sample_rate (归一化后加入向量)
    """

    duration_seconds: float = 0.0
    sample_rate: int = 16000

    # 频谱特征
    spectral_centroid_mean: float = 0.0
    spectral_bandwidth_mean: float = 0.0
    spectral_contrast_mean: List[float] = field(default_factory=lambda: [0.0] * 7)
    mfcc_means: List[float] = field(default_factory=lambda: [0.0] * 13)

    # 时域特征
    rms_mean: float = 0.0
    rms_std: float = 0.0
    zero_crossing_rate: float = 0.0

    # 语音特征
    speech_ratio: float = 0.0
    estimated_speaker_count: int = 1

    # 噪声特征
    noise_floor_db: float = -60.0
    snr_estimate: float = 20.0

    # 人类可读标签
    audio_signature: str = ""

    @property
    def dim(self) -> int:
        return 48

    def to_vector(self) -> np.ndarray:
        """转为归一化的 48 维向量

        维度顺序:
          0: duration_seconds
          1: sample_rate / 16000
          2: spectral_centroid_mean
          3: spectral_bandwidth_mean
          4-10: spectral_contrast_mean (7)
          11-23: mfcc_means (13)
          24: rms_mean
          25: rms_std
          26: zero_crossing_rate
          27: speech_ratio
          28: estimated_speaker_count
          29: noise_floor_db
          30: snr_estimate
          (后续维度保留扩展用)
        """
        vec = np.zeros(48, dtype=np.float32)
        vec[0] = self.duration_seconds
        vec[1] = self.sample_rate / 16000.0
        vec[2] = self.spectral_centroid_mean
        vec[3] = self.spectral_bandwidth_mean
        for i, v in enumerate(self.spectral_contrast_mean[:7]):
            vec[4 + i] = v
        for i, v in enumerate(self.mfcc_means[:13]):
            vec[11 + i] = v
        vec[24] = self.rms_mean
        vec[25] = self.rms_std
        vec[26] = self.zero_crossing_rate
        vec[27] = self.speech_ratio
        vec[28] = float(self.estimated_speaker_count)
        vec[29] = self.noise_floor_db
        vec[30] = self.snr_estimate
        return vec

    @staticmethod
    def from_vector(vec: np.ndarray) -> "AudioFingerprint":
        """从向量重建指纹"""
        return AudioFingerprint(
            duration_seconds=float(vec[0]),
            sample_rate=int(vec[1] * 16000),
            spectral_centroid_mean=float(vec[2]),
            spectral_bandwidth_mean=float(vec[3]),
            spectral_contrast_mean=[float(v) for v in vec[4:11]],
            mfcc_means=[float(v) for v in vec[11:24]],
            rms_mean=float(vec[24]),
            rms_std=float(vec[25]),
            zero_crossing_rate=float(vec[26]),
            speech_ratio=float(vec[27]),
            estimated_speaker_count=int(vec[28]),
            noise_floor_db=float(vec[29]),
            snr_estimate=float(vec[30]),
        )


# ---------------------------------------------------------------------------
# 马氏距离匹配器
# ---------------------------------------------------------------------------


class MahalanobisMatcher:
    """基于马氏距离的指纹匹配器

    通过协方差矩阵的逆对各维度进行"白化"，
    自动消除量纲和方差差异的影响。
    """

    def __init__(self):
        self._cov_matrix: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._dim: int = 48

    @property
    def is_fitted(self) -> bool:
        return self._cov_inv is not None

    def fit(self, fingerprint_vectors: np.ndarray):
        """用指纹库中的所有向量估计协方差矩阵和标准化参数

        Args:
            fingerprint_vectors: shape (N, 48) 的指纹向量矩阵
        """
        if len(fingerprint_vectors) < 2:
            logger.warning("Too few fingerprints to fit Mahalanobis matcher (need >= 2)")
            return

        self._mean = np.mean(fingerprint_vectors, axis=0)
        self._std = np.std(fingerprint_vectors, axis=0)

        # 标准化后估计协方差
        standardized = (fingerprint_vectors - self._mean) / (self._std + 1e-8)
        self._cov_matrix = np.cov(standardized.T)

        # 正则化：防止奇异矩阵
        reg = self._cov_matrix + np.eye(self._cov_matrix.shape[0]) * 1e-6
        try:
            self._cov_inv = np.linalg.inv(reg)
            logger.info(
                "MahalanobisMatcher fitted: %d vectors, dim=%d, cond=%.1f",
                len(fingerprint_vectors), self._dim,
                np.linalg.cond(reg),
            )
        except np.linalg.LinAlgError as e:
            logger.error("Failed to invert covariance matrix: %s", e)
            self._cov_inv = None

    def standardize(self, vec: np.ndarray) -> np.ndarray:
        """Z-score 标准化"""
        if self._mean is None or self._std is None:
            return vec
        return (vec - self._mean) / (self._std + 1e-8)

    def mahalanobis_distance(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """计算两个指纹向量之间的马氏距离

        Args:
            vec_a: 48 维向量
            vec_b: 48 维向量

        Returns:
            马氏距离（非负）
        """
        if not self.is_fitted:
            # 未拟合时回退到标准化欧氏距离
            return float(np.linalg.norm(vec_a - vec_b))

        a_std = self.standardize(vec_a)
        b_std = self.standardize(vec_b)
        delta = a_std - b_std
        return float(np.sqrt(delta.T @ self._cov_inv @ delta))

    def to_similarity(self, mah_dist: float, dim: int = 48) -> float:
        """将马氏距离转化为 [0, 1] 的相似度分数

        使用指数衰减映射: sim = exp(-dist / sqrt(dim))
        """
        return float(np.exp(-mah_dist / np.sqrt(dim)))

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """余弦相似度（作为备选方案）"""
        a_norm = vec_a / (np.linalg.norm(vec_a) + 1e-8)
        b_norm = vec_b / (np.linalg.norm(vec_b) + 1e-8)
        return float(np.dot(a_norm, b_norm))


# ---------------------------------------------------------------------------
# 音频指纹提取器
# ---------------------------------------------------------------------------


class AudioFingerprinter:
    """提取音频指纹并管理指纹→参数映射表

    使用 librosa 提取 48 维声学特征向量。
    存储于 SQLite 数据库中，支持 KNN 动态阈值匹配。
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        distance_method: str = "mahalanobis",
        knn_k: int = 3,
        min_absolute_similarity: float = 0.70,
        relative_margin: float = 0.08,
    ):
        """
        Args:
            db_path: SQLite 数据库路径，默认 ~/.vocal_subtitle/fingerprints.db
            distance_method: "mahalanobis" | "cosine"
            knn_k: 动态阈值 KNN 的 K 值
            min_absolute_similarity: 绝对下界
            relative_margin: 相对余量
        """
        if db_path is None:
            db_path = Path.home() / ".vocal_subtitle" / "fingerprints.db"
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._distance_method = distance_method
        self._knn_k = knn_k
        self._min_absolute_similarity = min_absolute_similarity
        self._relative_margin = relative_margin
        self._matcher = MahalanobisMatcher()

        self._init_db()

    # ------------------------------------------------------------------
    # 数据库初始化
    # ------------------------------------------------------------------

    def _init_db(self):
        """初始化 SQLite 数据库表结构"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fingerprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id TEXT NOT NULL,
                    vector_blob BLOB NOT NULL,
                    audio_hash TEXT NOT NULL,
                    audio_signature TEXT,
                    config_snapshot TEXT,
                    feedback_count INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(audio_hash, profile_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    profile_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    audio_hash TEXT NOT NULL,
                    alignment_coverage REAL,
                    diff_summary TEXT,
                    adjustments_json TEXT,
                    health_score_before REAL,
                    health_score_after REAL,
                    health_score_detail TEXT,
                    shadow_mode_run INTEGER DEFAULT 0,
                    asr_low_confidence_zones TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fp_profile
                ON fingerprints(profile_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fh_profile
                ON feedback_history(profile_id)
            """)
            conn.commit()

    def _get_conn(self) -> sqlite3.Connection:
        """获取数据库连接"""
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ------------------------------------------------------------------
    # 特征提取
    # ------------------------------------------------------------------

    def extract(self, audio_path: Path) -> Optional[AudioFingerprint]:
        """提取音频的 48 维声学特征向量

        Args:
            audio_path: 音频文件路径

        Returns:
            AudioFingerprint 或 None（提取失败时）
        """
        try:
            import librosa
        except ImportError:
            logger.warning("librosa not installed — using minimal fingerprint")
            return self._extract_minimal(audio_path)

        try:
            y, sr = librosa.load(str(audio_path), sr=16000, mono=True)
            duration = len(y) / sr

            fp = AudioFingerprint(
                duration_seconds=round(duration, 2),
                sample_rate=sr,
            )

            # ---- 频谱特征 ----
            # 频谱质心
            fp.spectral_centroid_mean = float(
                np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            )

            # 频谱带宽
            fp.spectral_bandwidth_mean = float(
                np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            )

            # 频谱对比度 (7维)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            fp.spectral_contrast_mean = [float(np.mean(c)) for c in contrast]

            # MFCC (13维)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            fp.mfcc_means = [float(np.mean(m)) for m in mfcc]

            # ---- 时域特征 ----
            fp.rms_mean = float(np.mean(librosa.feature.rms(y=y)))
            fp.rms_std = float(np.std(librosa.feature.rms(y=y)))
            fp.zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y)))

            # ---- 语音特征 ----
            # 语音占比：RMS > 20% 均值的帧视为语音
            rms = librosa.feature.rms(y=y)[0]
            threshold = np.mean(rms) * 0.2
            fp.speech_ratio = float(np.mean(rms > threshold))

            # ---- 噪声特征 ----
            # 噪声底：RMS 的下 10% 分位数
            noise_rms = np.percentile(rms, 10)
            fp.noise_floor_db = float(20 * np.log10(max(noise_rms, 1e-10)))

            # SNR 估计
            signal_rms = np.percentile(rms, 90)
            fp.snr_estimate = float(20 * np.log10(max(signal_rms / max(noise_rms, 1e-10), 1)))

            # ---- 音频签名 ----
            fp.audio_signature = self._describe(fp)

            logger.info(
                "Fingerprint extracted: %s (duration=%.1fs, snr=%.1fdB)",
                audio_path.name, fp.duration_seconds, fp.snr_estimate,
            )
            return fp

        except Exception as e:
            logger.warning("Fingerprint extraction failed for %s: %s", audio_path, e)
            return self._extract_minimal(audio_path)

    def _extract_minimal(self, audio_path: Path) -> Optional[AudioFingerprint]:
        """无 librosa 时的最小指纹提取（使用 soundfile）"""
        try:
            import soundfile as sf
            y, sr = sf.read(str(audio_path), dtype="float32")
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            duration = len(y) / sr

            fp = AudioFingerprint(
                duration_seconds=round(duration, 2),
                sample_rate=sr,
                rms_mean=float(np.sqrt(np.mean(y ** 2))),
                rms_std=float(np.std(np.sqrt(np.mean(
                    np.array([y[i:i+1024] for i in range(0, len(y), 1024) if i+1024 <= len(y)]) ** 2,
                    axis=1,
                )))) if len(y) >= 1024 else 0.0,
                zero_crossing_rate=float(np.mean(np.abs(np.diff(np.sign(y))))) / 2,
                speech_ratio=float(np.mean(np.abs(y) > np.mean(np.abs(y)) * 0.2)),
            )
            fp.audio_signature = self._describe(fp)
            return fp
        except Exception as e:
            logger.error("Minimal fingerprint also failed for %s: %s", audio_path, e)
            return None

    @staticmethod
    def _describe(fp: AudioFingerprint) -> str:
        """生成人类可读的音频描述"""
        parts = []
        if fp.speech_ratio > 0.6:
            parts.append("高语音密度")
        elif fp.speech_ratio > 0.3:
            parts.append("中等语音密度")
        else:
            parts.append("低语音密度")

        if fp.snr_estimate > 25:
            parts.append("高信噪比(录音棚)")
        elif fp.snr_estimate > 15:
            parts.append("中等信噪比(一般环境)")
        else:
            parts.append("低信噪比(嘈杂环境)")

        if fp.duration_seconds < 60:
            parts.append("短音频")
        elif fp.duration_seconds < 600:
            parts.append("中等长度")
        else:
            parts.append("长音频")

        return " - ".join(parts)

    def to_vector(self, fp: AudioFingerprint) -> np.ndarray:
        """转为归一化的 48 维向量"""
        return fp.to_vector()

    # ------------------------------------------------------------------
    # 音频哈希
    # ------------------------------------------------------------------

    @staticmethod
    def compute_audio_hash(audio_path: Path) -> str:
        """计算音频文件的 SHA256 哈希"""
        sha = hashlib.sha256()
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()

    # ------------------------------------------------------------------
    # 指纹存储 CRUD
    # ------------------------------------------------------------------

    def store(
        self,
        profile_id: str,
        fingerprint: AudioFingerprint,
        audio_hash: str,
        config_snapshot: Optional[Dict[str, Any]] = None,
    ) -> int:
        """存储指纹到数据库

        Returns:
            row id
        """
        vector = fingerprint.to_vector()
        vector_blob = vector.tobytes()
        now = datetime.now().isoformat()

        config_json = json.dumps(config_snapshot) if config_snapshot else "{}"

        with self._get_conn() as conn:
            # UPSERT (INSERT OR REPLACE on conflict)
            existing = conn.execute(
                "SELECT id, feedback_count FROM fingerprints WHERE audio_hash = ? AND profile_id = ?",
                (audio_hash, profile_id),
            ).fetchone()

            if existing:
                new_count = existing[1] + 1
                conn.execute(
                    """UPDATE fingerprints
                       SET vector_blob = ?, config_snapshot = ?,
                           feedback_count = ?, updated_at = ?
                       WHERE id = ?""",
                    (vector_blob, config_json, new_count, now, existing[0]),
                )
                conn.commit()
                logger.info("Fingerprint updated: id=%d, feedback_count=%d", existing[0], new_count)
                return existing[0]
            else:
                cursor = conn.execute(
                    """INSERT INTO fingerprints
                       (profile_id, vector_blob, audio_hash, audio_signature,
                        config_snapshot, feedback_count, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, 1, ?, ?)""",
                    (
                        profile_id, vector_blob, audio_hash,
                        fingerprint.audio_signature, config_json,
                        now, now,
                    ),
                )
                conn.commit()
                logger.info("Fingerprint stored: id=%d", cursor.lastrowid)
                return cursor.lastrowid

    def load_all_vectors(self) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """加载所有指纹向量及其元数据"""
        vectors = []
        metadata = []
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, vector_blob, profile_id, audio_hash, audio_signature, "
                "config_snapshot, feedback_count FROM fingerprints"
            ).fetchall()

        for row in rows:
            vec = np.frombuffer(row[1], dtype=np.float32)
            if len(vec) == 48:
                vectors.append(vec)
                metadata.append({
                    "id": row[0],
                    "profile_id": row[2],
                    "audio_hash": row[3],
                    "audio_signature": row[4],
                    "config_snapshot": json.loads(row[5]) if row[5] else {},
                    "feedback_count": row[6],
                })

        return vectors, metadata

    def get_by_audio_hash(
        self, audio_hash: str, profile_id: str = "user_default"
    ) -> Optional[Dict[str, Any]]:
        """按音频哈希查找指纹"""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM fingerprints WHERE audio_hash = ? AND profile_id = ?",
                (audio_hash, profile_id),
            ).fetchone()

        if not row:
            return None

        return {
            "id": row[0],
            "profile_id": row[1],
            "vector": np.frombuffer(row[2], dtype=np.float32),
            "audio_hash": row[3],
            "audio_signature": row[4],
            "config_snapshot": json.loads(row[5]) if row[5] else {},
            "feedback_count": row[6],
            "created_at": row[7],
            "updated_at": row[8],
        }

    def delete_by_id(self, fingerprint_id: int) -> bool:
        """删除指定指纹"""
        with self._get_conn() as conn:
            cursor = conn.execute("DELETE FROM fingerprints WHERE id = ?", (fingerprint_id,))
            conn.commit()
            return cursor.rowcount > 0

    def count(self) -> int:
        """返回指纹总数"""
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM fingerprints").fetchone()
            return row[0] if row else 0

    def list_all(self) -> List[Dict[str, Any]]:
        """列出所有指纹"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id, profile_id, audio_hash, audio_signature, "
                "feedback_count, created_at FROM fingerprints ORDER BY updated_at DESC"
            ).fetchall()

        return [
            {
                "id": r[0],
                "profile_id": r[1],
                "audio_hash": r[2],
                "audio_signature": r[3],
                "feedback_count": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # 指纹匹配
    # ------------------------------------------------------------------

    def find_similar(
        self,
        fingerprint: AudioFingerprint,
    ) -> Optional[Tuple[str, float]]:
        """在指纹库中查找最近邻（动态阈值）

        Args:
            fingerprint: 待匹配的音频指纹

        Returns:
            (profile_id, confidence) 或 None
        """
        vectors, metadata = self.load_all_vectors()
        if not vectors:
            return None

        query_vec = fingerprint.to_vector()

        # 计算所有距离/相似度
        similarities = []
        for vec in vectors:
            if self._distance_method == "mahalanobis" and self._matcher.is_fitted:
                dist = self._matcher.mahalanobis_distance(query_vec, vec)
                sim = self._matcher.to_similarity(dist)
            else:
                sim = self._matcher.cosine_similarity(query_vec, vec)
            similarities.append(sim)

        # 按相似度降序排序
        indexed = sorted(
            enumerate(similarities),
            key=lambda x: x[1],
            reverse=True,
        )

        if not indexed:
            return None

        # KNN 动态阈值
        top_k = indexed[:self._knn_k]
        top1_idx, top1_sim = top_k[0]
        baseline = float(np.mean([s for _, s in top_k]))

        # 匹配条件
        if top1_sim < self._min_absolute_similarity:
            logger.debug(
                "Fingerprint match failed: top1_sim=%.3f < min=%.2f",
                top1_sim, self._min_absolute_similarity,
            )
            return None

        if top1_sim < baseline + self._relative_margin:
            logger.debug(
                "Fingerprint match failed: top1_sim=%.3f < baseline+margin=%.3f",
                top1_sim, baseline + self._relative_margin,
            )
            return None

        # Top-1 与 Top-2 差距检查
        if len(top_k) >= 2:
            top2_sim = top_k[1][1]
            if top1_sim - top2_sim < 0.05:
                logger.debug(
                    "Fingerprint match ambiguous: top1=%.3f vs top2=%.3f (gap=%.3f)",
                    top1_sim, top2_sim, top1_sim - top2_sim,
                )
                return None

        profile_id = metadata[top1_idx]["profile_id"]
        logger.info(
            "Fingerprint matched: profile=%s, confidence=%.3f",
            profile_id, top1_sim,
        )
        return (profile_id, top1_sim)

    def refit_matcher(self):
        """用当前指纹库数据重新拟合马氏距离匹配器"""
        vectors, _ = self.load_all_vectors()
        if len(vectors) >= 2:
            self._matcher.fit(np.array(vectors))
            logger.info("MahalanobisMatcher refitted with %d vectors", len(vectors))
        else:
            logger.info("Not enough vectors to refit matcher (%d)", len(vectors))

    # ------------------------------------------------------------------
    # 反馈历史
    # ------------------------------------------------------------------

    def record_feedback(
        self,
        profile_id: str,
        audio_hash: str,
        alignment_coverage: float,
        diff_summary: str,
        adjustments: Dict[str, Any],
        health_before: Optional[float] = None,
        health_after: Optional[float] = None,
        health_detail: Optional[Dict[str, float]] = None,
        shadow_mode: bool = False,
        asr_low_confidence_zones: Optional[List[Dict]] = None,
    ) -> int:
        """记录反馈历史到数据库"""
        now = datetime.now().isoformat()
        with self._get_conn() as conn:
            cursor = conn.execute(
                """INSERT INTO feedback_history
                   (profile_id, timestamp, audio_hash, alignment_coverage,
                    diff_summary, adjustments_json, health_score_before,
                    health_score_after, health_score_detail, shadow_mode_run,
                    asr_low_confidence_zones)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    profile_id, now, audio_hash,
                    round(alignment_coverage, 4) if alignment_coverage else None,
                    diff_summary,
                    json.dumps(adjustments),
                    round(health_before, 2) if health_before is not None else None,
                    round(health_after, 2) if health_after is not None else None,
                    json.dumps(health_detail) if health_detail else None,
                    1 if shadow_mode else 0,
                    json.dumps(asr_low_confidence_zones) if asr_low_confidence_zones else None,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_feedback_history(
        self,
        profile_id: str = "user_default",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """获取反馈历史"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM feedback_history
                   WHERE profile_id = ?
                   ORDER BY timestamp DESC LIMIT ?""",
                (profile_id, limit),
            ).fetchall()

        return [
            {
                "id": r[0],
                "profile_id": r[1],
                "timestamp": r[2],
                "audio_hash": r[3],
                "alignment_coverage": r[4],
                "diff_summary": r[5],
                "adjustments": json.loads(r[6]) if r[6] else {},
                "health_score_before": r[7],
                "health_score_after": r[8],
                "health_score_detail": json.loads(r[9]) if r[9] else None,
                "shadow_mode_run": bool(r[10]),
                "asr_low_confidence_zones": json.loads(r[11]) if r[11] else None,
            }
            for r in rows
        ]

    def get_health_trend(
        self, profile_id: str = "user_default", limit: int = 20
    ) -> List[Dict[str, Any]]:
        """获取健康度趋势数据（用于前端趋势图）"""
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT timestamp, health_score_before, health_score_after,
                           shadow_mode_run, diff_summary
                   FROM feedback_history
                   WHERE profile_id = ? AND health_score_after IS NOT NULL
                   ORDER BY timestamp ASC LIMIT ?""",
                (profile_id, limit),
            ).fetchall()

        return [
            {
                "timestamp": r[0],
                "health_before": r[1],
                "health_after": r[2],
                "shadow_mode": bool(r[3]),
                "summary": r[4],
            }
            for r in rows
        ]
