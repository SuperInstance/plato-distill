use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationRecord {
    pub from_level: u8,
    pub to_level: u8,
    pub input_tile: String,
    pub teacher_output: String,
    pub student_output: String,
    pub loss: f64,
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationConfig {
    pub teacher_level: u8,
    pub student_level: u8,
    pub temperature: f64,
    pub top_k: usize,
    pub min_samples: usize,
}

impl Default for DistillationConfig {
    fn default() -> Self {
        Self {
            teacher_level: 4,
            student_level: 2,
            temperature: 2.0,
            top_k: 5,
            min_samples: 10,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationStats {
    pub total_records: u64,
    pub avg_loss: f64,
    pub loss_trend: TrendDirection,
    pub samples_by_level: HashMap<u8, u64>,
}

impl Default for DistillationStats {
    fn default() -> Self {
        Self {
            total_records: 0,
            avg_loss: 0.0,
            loss_trend: TrendDirection::Stable,
            samples_by_level: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeTransfer {
    pub source: String,
    pub target: String,
    pub accuracy_gain: f64,
    pub samples_used: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationPipeline {
    pub config: DistillationConfig,
    pub history: Vec<DistillationRecord>,
    pub stats: DistillationStats,
}

impl DistillationPipeline {
    pub fn new(config: DistillationConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
            stats: DistillationStats::default(),
        }
    }

    pub fn record(&mut self, record: DistillationRecord) {
        let loss = record.loss;
        let to_level = record.to_level;

        self.stats.total_records += 1;
        let n = self.stats.total_records as f64;
        self.stats.avg_loss = self.stats.avg_loss * ((n - 1.0) / n) + loss / n;

        *self.stats.samples_by_level.entry(to_level).or_insert(0) += 1;

        self.history.push(record);
    }

    pub fn compute_loss(teacher: &str, student: &str) -> f64 {
        if teacher.is_empty() && student.is_empty() {
            return 0.0;
        }
        let max_len = teacher.len().max(student.len()) as f64;
        if max_len == 0.0 {
            return 0.0;
        }
        let teacher_lower = teacher.to_lowercase();
        let student_lower = student.to_lowercase();
        let distance = levenshtein_distance(&teacher_lower, &student_lower);
        distance as f64 / max_len
    }

    pub fn compute_kl_divergence(teacher_probs: &[f64], student_probs: &[f64]) -> f64 {
        let len = teacher_probs.len().min(student_probs.len());
        let mut kl = 0.0;
        for i in 0..len {
            let p = teacher_probs[i];
            let q = student_probs[i];
            if p > 0.0 && q > 0.0 {
                kl += p * (p / q).ln();
            }
        }
        kl
    }

    pub fn stats(&self) -> &DistillationStats {
        &self.stats
    }

    pub fn trend(&self, window: usize) -> TrendDirection {
        if self.history.len() < 2 || window < 2 {
            return TrendDirection::Stable;
        }
        let w = window.min(self.history.len());
        let recent: Vec<f64> = self.history.iter().rev().take(w).map(|r| r.loss).collect();

        let mid = w / 2;
        let first_half: f64 = recent[mid..].iter().sum::<f64>() / (w - mid) as f64;
        let second_half: f64 = recent[..mid].iter().sum::<f64>() / mid as f64;

        let diff = first_half - second_half;
        let threshold = 0.05 * first_half.max(0.01);

        if diff > threshold {
            TrendDirection::Improving
        } else if diff < -threshold {
            TrendDirection::Degrading
        } else {
            TrendDirection::Stable
        }
    }

    pub fn best_student_for(&self, level: u8) -> Option<DistillationRecord> {
        self.history
            .iter()
            .filter(|r| r.to_level == level)
            .min_by(|a, b| a.loss.partial_cmp(&b.loss).unwrap_or(std::cmp::Ordering::Equal))
            .cloned()
    }

    pub fn knowledge_transfers(&self) -> Vec<KnowledgeTransfer> {
        let mut transfers: HashMap<(u8, u8), (f64, usize)> = HashMap::new();

        for record in &self.history {
            let key = (record.from_level, record.to_level);
            let entry = transfers.entry(key).or_insert((0.0, 0));
            entry.0 += record.loss;
            entry.1 += 1;
        }

        transfers
            .into_iter()
            .map(|((from, to), (total_loss, count))| {
                let avg_loss = total_loss / count as f64;
                let accuracy_gain = 1.0 - avg_loss;
                KnowledgeTransfer {
                    source: format!("L{}", from),
                    target: format!("L{}", to),
                    accuracy_gain,
                    samples_used: count,
                }
            })
            .collect()
    }

    pub fn is_ready_for_promotion(&self, level: u8, threshold: f64) -> bool {
        let count = self.stats.samples_by_level.get(&level).copied().unwrap_or(0);
        if (count as usize) < self.config.min_samples {
            return false;
        }
        self.best_student_for(level)
            .map(|r| r.loss <= threshold)
            .unwrap_or(false)
    }

    /// Apply temperature scaling to logits, converting to probabilities.
    pub fn temperature_scale(&self, logits: &[f64]) -> Vec<f64> {
        let scaled: Vec<f64> = logits.iter().map(|&l| (l / self.config.temperature).exp()).collect();
        let sum: f64 = scaled.iter().sum();
        if sum == 0.0 {
            return vec![0.0; logits.len()];
        }
        scaled.iter().map(|&v| v / sum).collect()
    }
}

fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (n, m) = (a.len(), b.len());

    if n == 0 {
        return m;
    }
    if m == 0 {
        return n;
    }

    let mut prev: Vec<usize> = (0..=m).collect();
    let mut curr: Vec<usize> = vec![0; m + 1];

    for i in 1..=n {
        curr[0] = i;
        for j in 1..=m {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[m]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(from: u8, to: u8, loss: f64, ts: u64) -> DistillationRecord {
        DistillationRecord {
            from_level: from,
            to_level: to,
            input_tile: "tile".into(),
            teacher_output: "teacher".into(),
            student_output: "student".into(),
            loss,
            timestamp: ts,
        }
    }

    #[test]
    fn test_record_creation_and_storage() {
        let mut p = DistillationPipeline::new(DistillationConfig::default());
        p.record(make_record(4, 2, 0.3, 1));
        p.record(make_record(4, 2, 0.2, 2));
        assert_eq!(p.history.len(), 2);
        assert_eq!(p.stats.total_records, 2);
    }

    #[test]
    fn test_loss_identical_strings() {
        let loss = DistillationPipeline::compute_loss("hello world", "hello world");
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_loss_different_strings() {
        let loss = DistillationPipeline::compute_loss("hello", "world");
        assert!(loss > 0.0);
    }

    #[test]
    fn test_loss_empty_strings() {
        let loss = DistillationPipeline::compute_loss("", "");
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_kl_divergence_identical() {
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let kl = DistillationPipeline::compute_kl_divergence(&probs, &probs);
        assert!(kl.abs() < 1e-10);
    }

    #[test]
    fn test_kl_divergence_different() {
        let p = vec![0.5, 0.5];
        let q = vec![0.9, 0.1];
        let kl = DistillationPipeline::compute_kl_divergence(&p, &q);
        assert!(kl > 0.0);
    }

    #[test]
    fn test_trend_improving() {
        let mut p = DistillationPipeline::new(DistillationConfig::default());
        for i in 0..10 {
            p.record(make_record(4, 2, 1.0 - (i as f64 * 0.09), i));
        }
        assert_eq!(p.trend(10), TrendDirection::Improving);
    }

    #[test]
    fn test_trend_stable() {
        let mut p = DistillationPipeline::new(DistillationConfig::default());
        for i in 0..10 {
            p.record(make_record(4, 2, 0.5, i));
        }
        assert_eq!(p.trend(10), TrendDirection::Stable);
    }

    #[test]
    fn test_trend_degrading() {
        let mut p = DistillationPipeline::new(DistillationConfig::default());
        for i in 0..10 {
            p.record(make_record(4, 2, 0.1 + (i as f64 * 0.09), i));
        }
        assert_eq!(p.trend(10), TrendDirection::Degrading);
    }

    #[test]
    fn test_best_student_for() {
        let mut p = DistillationPipeline::new(DistillationConfig::default());
        p.record(make_record(4, 2, 0.5, 1));
        p.record(make_record(4, 2, 0.2, 2));
        p.record(make_record(4, 2, 0.8, 3));
        let best = p.best_student_for(2).unwrap();
        assert!((best.loss - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_best_student_missing_level() {
        let p = DistillationPipeline::new(DistillationConfig::default());
        assert!(p.best_student_for(0).is_none());
    }

    #[test]
    fn test_knowledge_transfers() {
        let mut p = DistillationPipeline::new(DistillationConfig::default());
        p.record(make_record(4, 2, 0.2, 1));
        p.record(make_record(4, 2, 0.4, 2));
        p.record(make_record(3, 1, 0.1, 3));
        let transfers = p.knowledge_transfers();
        assert_eq!(transfers.len(), 2);
        let t42 = transfers.iter().find(|t| t.source == "L4" && t.target == "L2").unwrap();
        assert!((t42.accuracy_gain - 0.7).abs() < 1e-10);
        assert_eq!(t42.samples_used, 2);
    }

    #[test]
    fn test_promotion_ready() {
        let mut p = DistillationPipeline::new(DistillationConfig {
            min_samples: 3,
            ..Default::default()
        });
        for i in 0..5 {
            p.record(make_record(4, 2, 0.1, i));
        }
        assert!(p.is_ready_for_promotion(2, 0.2));
        assert!(!p.is_ready_for_promotion(2, 0.05));
    }

    #[test]
    fn test_promotion_not_enough_samples() {
        let mut p = DistillationPipeline::new(DistillationConfig {
            min_samples: 10,
            ..Default::default()
        });
        p.record(make_record(4, 2, 0.01, 1));
        assert!(!p.is_ready_for_promotion(2, 0.5));
    }

    #[test]
    fn test_stats_accuracy() {
        let mut p = DistillationPipeline::new(DistillationConfig::default());
        p.record(make_record(4, 2, 0.4, 1));
        p.record(make_record(4, 2, 0.6, 2));
        let stats = p.stats();
        assert_eq!(stats.total_records, 2);
        assert!((stats.avg_loss - 0.5).abs() < 1e-10);
        assert_eq!(*stats.samples_by_level.get(&2).unwrap(), 2);
    }

    #[test]
    fn test_temperature_scaling() {
        let p = DistillationPipeline::new(DistillationConfig {
            temperature: 1.0,
            ..Default::default()
        });
        let probs = p.temperature_scale(&[1.0, 1.0, 1.0]);
        assert!(probs.iter().all(|&v| (v - 1.0 / 3.0).abs() < 1e-10));
    }

    #[test]
    fn test_empty_pipeline() {
        let p = DistillationPipeline::new(DistillationConfig::default());
        assert_eq!(p.history.len(), 0);
        assert_eq!(p.stats.total_records, 0);
        assert_eq!(p.trend(5), TrendDirection::Stable);
        assert!(p.best_student_for(0).is_none());
        assert!(p.knowledge_transfers().is_empty());
        assert!(!p.is_ready_for_promotion(0, 0.5));
    }

    #[test]
    fn test_single_record() {
        let mut p = DistillationPipeline::new(DistillationConfig {
            min_samples: 1,
            ..Default::default()
        });
        p.record(make_record(4, 2, 0.3, 1));
        assert_eq!(p.history.len(), 1);
        assert!((p.stats.avg_loss - 0.3).abs() < 1e-10);
        assert_eq!(p.trend(1), TrendDirection::Stable);
    }

    #[test]
    fn test_all_same_loss_trend() {
        let mut p = DistillationPipeline::new(DistillationConfig::default());
        for i in 0..10 {
            p.record(make_record(4, 2, 0.5, i));
        }
        assert_eq!(p.trend(10), TrendDirection::Stable);
    }

    #[test]
    fn test_serialization() {
        let mut p = DistillationPipeline::new(DistillationConfig::default());
        p.record(make_record(4, 2, 0.3, 1));
        let json = serde_json::to_string(&p).unwrap();
        let deserialized: DistillationPipeline = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.history.len(), 1);
        assert!((deserialized.history[0].loss - 0.3).abs() < 1e-10);
    }
}
