# plato-distill

> Progressive knowledge distillation tracking for PLATO rooms

## What This Does

plato-distill tracks the process of distilling knowledge from teacher models to student models across PLATO rooms. It records every distillation step, computes loss trends, and determines when a student model is ready for promotion — replacing its teacher with a smaller, faster version that retains the essential knowledge.

## The Key Idea

Knowledge distillation is like a master craftsman training an apprentice. The master (teacher model) is large and slow but wise. The apprentice (student model) is small and fast but inexperienced. Through training, the apprentice learns to produce outputs that match the master's. plato-distill tracks this process: recording each lesson, measuring how close the apprentice gets, and signaling when they're ready to work independently.

## Install

```bash
cargo add plato-distill
```

## Quick Start

```rust
use plato_distill::{DistillationPipeline, DistillationConfig, DistillationRecord};

let mut pipeline = DistillationPipeline::new(DistillationConfig::default());

// Record a distillation step
pipeline.record(DistillationRecord {
    from_level: 4,
    to_level: 2,
    input_tile: "sensor_reading".into(),
    teacher_output: "alert: thermal spike".into(),
    student_output: "alert: thermal".into(),
    loss: DistillationPipeline::compute_loss("alert: thermal spike", "alert: thermal"),
    timestamp: 1700000000,
});

// Check progress
println!("Avg loss: {:.3}", pipeline.stats().avg_loss);
println!("Trend: {:?}", pipeline.trend(10));
println!("Ready for promotion: {}", pipeline.is_ready_for_promotion(2, 0.2));
```

## API Reference

### Core Types

| Type | Description |
|---|---|
| `DistillationConfig { teacher_level, student_level, temperature, top_k, min_samples }` | Config (defaults: 4→2, temp=2.0, top_k=5, min_samples=10) |
| `DistillationRecord { from_level, to_level, input_tile, teacher_output, student_output, loss, timestamp }` | One distillation step |
| `DistillationStats { total_records, avg_loss, loss_trend, samples_by_level }` | Running statistics |
| `KnowledgeTransfer { source, target, accuracy_gain, samples_used }` | Aggregated transfer metrics |
| `TrendDirection` | `Improving` / `Stable` / `Degrading` |

### DistillationPipeline

```rust
let mut p = DistillationPipeline::new(config);
p.record(record);                          // Record a step
p.stats();                                 // → &DistillationStats
p.trend(10);                               // → TrendDirection over last 10
p.best_student_for(level);                 // → Option<DistillationRecord>
p.knowledge_transfers();                   // → Vec<KnowledgeTransfer>
p.is_ready_for_promotion(level, threshold);// → bool
p.temperature_scale(&[1.0, 2.0, 3.0]);    // → Vec<f64> (softmax with temperature)
```

### Loss Functions

```rust
// String-based loss (Levenshtein distance / max length)
DistillationPipeline::compute_loss("hello world", "hello worl");

// KL divergence between probability distributions
DistillationPipeline::compute_kl_divergence(&teacher_probs, &student_probs);
```

## How It Works

Loss is computed as normalized Levenshtein distance between teacher and student outputs. The trend detector splits recent history in half and compares average loss — if the recent half is significantly lower, trend is Improving. Temperature scaling applies softmax with a temperature parameter (higher = more uniform distribution).

## Testing

20 tests: record creation, string loss, KL divergence, trend detection (improving/stable/degrading), best student selection, knowledge transfer aggregation, promotion readiness, temperature scaling, serialization.

## License

Apache-2.0
