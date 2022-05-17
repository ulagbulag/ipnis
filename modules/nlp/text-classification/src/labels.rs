#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Labels {
    pub contradiction: Option<usize>,
    pub entailment: Option<usize>,
    pub neutral: Option<usize>,
}

impl Default for Labels {
    fn default() -> Self {
        Self::roberta()
    }
}

impl Labels {
    pub fn roberta() -> Self {
        Self {
            contradiction: Some(0),
            entailment: Some(1),
            neutral: Some(2),
        }
    }
}
