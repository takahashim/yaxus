//! The errors of the API as HTTP responses, and the JSON body extractor that reports a
//! malformed request as one of them.

use axum::Json;
use axum::extract::{FromRequest, Request};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde::de::DeserializeOwned;
use yaxus_api::{ApiError, ErrorCode, ErrorResponse};

use crate::queue::Rejected;

/// An [`ApiError`] answered with the status of its code.
#[derive(Debug)]
pub struct ApiFailure(pub ApiError);

impl ApiFailure {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        ApiFailure(ApiError::new(code, message))
    }

    /// The inference thread has stopped.
    pub fn worker_stopped() -> Self {
        ApiFailure::new(ErrorCode::Unavailable, "inference worker stopped")
    }
}

impl From<ApiError> for ApiFailure {
    fn from(e: ApiError) -> Self {
        ApiFailure(e)
    }
}

impl From<Rejected> for ApiFailure {
    fn from(r: Rejected) -> Self {
        match r {
            Rejected::Full => ApiFailure::new(ErrorCode::QueueFull, "translation queue is full; retry later"),
            Rejected::Stopped => ApiFailure::worker_stopped(),
        }
    }
}

/// The HTTP status of each error code.
fn status_of(code: ErrorCode) -> StatusCode {
    match code {
        ErrorCode::InvalidRequest | ErrorCode::PromptTooLong => StatusCode::BAD_REQUEST,
        ErrorCode::Unauthorized => StatusCode::UNAUTHORIZED,
        ErrorCode::QueueFull => StatusCode::TOO_MANY_REQUESTS,
        ErrorCode::Unavailable => StatusCode::SERVICE_UNAVAILABLE,
        ErrorCode::Internal => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

impl IntoResponse for ApiFailure {
    fn into_response(self) -> Response {
        (status_of(self.0.code), Json(ErrorResponse { error: self.0 })).into_response()
    }
}

/// A JSON body; one that cannot be parsed is an `invalid_request`.
pub struct ApiJson<T>(pub T);

impl<S: Send + Sync, T: DeserializeOwned> FromRequest<S> for ApiJson<T> {
    type Rejection = ApiFailure;

    async fn from_request(req: Request, state: &S) -> Result<Self, ApiFailure> {
        match Json::<T>::from_request(req, state).await {
            Ok(Json(v)) => Ok(ApiJson(v)),
            Err(e) => Err(ApiFailure::new(ErrorCode::InvalidRequest, e.body_text())),
        }
    }
}
