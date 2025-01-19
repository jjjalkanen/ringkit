#[cfg(feature = "verify")]
pub trait DebugIfVerify: std::fmt::Debug {}

#[cfg(feature = "verify")]
impl<T: std::fmt::Debug> DebugIfVerify for T {}

#[cfg(not(feature = "verify"))]
pub trait DebugIfVerify {}

#[cfg(not(feature = "verify"))]
impl<T> DebugIfVerify for T {}
