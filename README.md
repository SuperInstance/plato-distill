# plato-distill

Progressive knowledge distillation tracking for PLATO rooms.

## Overview

Tracks teacher→student distillation across abstraction levels with:

- **DistillationPipeline** — records distillation attempts, computes running stats, detects trends
- **Loss computation** — normalized Levenshtein distance between teacher/student outputs
- **KL divergence** — between teacher and student probability distributions
- **Temperature scaling** — softmax with configurable temperature
- **Trend detection** — improving / stable / degrading loss trends
- **Promotion readiness** — check if a student level is ready for promotion based on sample count and loss threshold

## Usage

```rust
use plato_distill::*;

let mut pipeline = DistillationPipeline::new(DistillationConfig::default());
pipeline.record(DistillationRecord {
    from_level: 4,
    to_level: 2,
    input_tile: "tile-42".into(),
    teacher_output: "correct answer".into(),
    student_output: "correct answer".into(),
    loss: 0.05,
    timestamp: 1000,
});
println!("trend: {:?}", pipeline.trend(10));
println!("ready for promotion: {}", pipeline.is_ready_for_promotion(2, 0.1));
```

## License

Apache-2.0
