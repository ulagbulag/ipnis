use ipnis_common::onnxruntime::GraphOptimizationLevel;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EngineConfig {
    pub optimization_level: GraphOptimizationLevel,
    pub number_threads: u8,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            optimization_level: GraphOptimizationLevel::Basic,
            number_threads: 1,
        }
    }
}
