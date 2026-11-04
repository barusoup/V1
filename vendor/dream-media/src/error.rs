use std::fmt;

/// Everything this crate can fail with.
#[derive(Debug)]
pub enum Error {
    /// A file or process-pipe operation failed.
    Io(std::io::Error),
    /// A decoder, probe or configuration problem, with a user-facing message.
    Message(String),
}

impl Error {
    pub(crate) fn msg(message: impl Into<String>) -> Self {
        Self::Message(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "{error}"),
            Self::Message(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Message(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<image::ImageError> for Error {
    fn from(error: image::ImageError) -> Self {
        Self::Message(format!("decode image: {error}"))
    }
}

/// Crate result type.
pub type Result<T> = std::result::Result<T, Error>;
