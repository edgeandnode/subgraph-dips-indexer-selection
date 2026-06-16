//! Domain newtypes that make illegal states unrepresentable.

use std::fmt;

/// An indexer's on-chain address, normalised to lowercase so equality and set
/// membership behave like the Python service (which lowercases on ingest).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct IndexerId(String);

impl IndexerId {
    pub fn new(s: impl Into<String>) -> Self {
        IndexerId(s.into().to_lowercase())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for IndexerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A subgraph deployment id (IPFS hash). Case-sensitive — base58 hashes are not
/// lowercased.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DeploymentId(String);

impl DeploymentId {
    pub fn new(s: impl Into<String>) -> Self {
        DeploymentId(s.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for DeploymentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A normalised utility, guaranteed to live in `[0, 1]` where 1 is best.
///
/// The inner value is private, so the only ways to obtain a `Utility` are the
/// clamping/validating constructors below. There is no path by which a metric
/// can "fill up" past 1.0 — the bug behind Finding 1's zero-fee → 1.0 fill is
/// simply not expressible.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Utility(f64);

impl Utility {
    pub const ZERO: Utility = Utility(0.0);
    pub const ONE: Utility = Utility(1.0);

    /// Clamp any finite value into `[0, 1]`; NaN maps to 0 (conservative).
    pub fn clamp(x: f64) -> Utility {
        if x.is_nan() {
            Utility(0.0)
        } else {
            Utility(x.clamp(0.0, 1.0))
        }
    }

    /// Strict constructor: rejects anything outside `[0, 1]` (or non-finite).
    pub fn try_new(x: f64) -> Result<Utility, DomainError> {
        if x.is_finite() && (0.0..=1.0).contains(&x) {
            Ok(Utility(x))
        } else {
            Err(DomainError::UtilityOutOfRange(x))
        }
    }

    pub fn get(self) -> f64 {
        self.0
    }

    pub fn is_zero(self) -> bool {
        self.0 == 0.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum DomainError {
    UtilityOutOfRange(f64),
}

impl fmt::Display for DomainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DomainError::UtilityOutOfRange(x) => {
                write!(f, "utility {x} is outside the [0, 1] range")
            }
        }
    }
}

impl std::error::Error for DomainError {}
