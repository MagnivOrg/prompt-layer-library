class PromptLayerError(Exception):
    """Base exception for all PromptLayer SDK errors."""

    def __init__(self, message: str, response=None, body=None):
        super().__init__(message)
        self.message = message
        self.response = response
        self.body = body

    def __str__(self):
        return self.message


class PromptLayerAPIError(PromptLayerError):
    """Base exception for API-related errors."""

    pass


class PromptLayerBadRequestError(PromptLayerAPIError):
    """Exception raised for 400 Bad Request errors.

    Indicates that the request was malformed or contained invalid parameters.
    """

    pass


class PromptLayerAuthenticationError(PromptLayerAPIError):
    """Exception raised for 401 Unauthorized errors.

    Indicates that the API key is missing, invalid, or expired.
    """

    pass


class PromptLayerPermissionDeniedError(PromptLayerAPIError):
    """Exception raised for 403 Forbidden errors.

    Indicates that the API key doesn't have permission to perform the requested operation.
    """

    pass


class PromptLayerNotFoundError(PromptLayerAPIError):
    """Exception raised for 404 Not Found errors.

    Indicates that the requested resource (e.g., prompt template) was not found.
    """

    pass


class PromptLayerConflictError(PromptLayerAPIError):
    """Exception raised for 409 Conflict errors.

    Indicates that the request conflicts with the current state of the resource.
    """

    pass


class PromptLayerUnprocessableEntityError(PromptLayerAPIError):
    """Exception raised for 422 Unprocessable Entity errors.

    Indicates that the request was well-formed but contains semantic errors.
    """

    pass


class PromptLayerRateLimitError(PromptLayerAPIError):
    """Exception raised for 429 Too Many Requests errors.

    Indicates that the API rate limit has been exceeded.
    """

    pass


class PromptLayerInternalServerError(PromptLayerAPIError):
    """Exception raised for 500+ Internal Server errors.

    Indicates that the PromptLayer API encountered an internal error.
    """

    pass


class PromptLayerAPIStatusError(PromptLayerAPIError):
    """Exception raised for other API errors not covered by specific exception classes."""

    pass


class PromptLayerAPIConnectionError(PromptLayerError):
    """Exception raised when unable to connect to the API.

    This can be due to network issues, timeouts, or connection errors.
    """

    pass


class PromptLayerAPITimeoutError(PromptLayerError):
    """Exception raised when an API request times out."""

    pass


class PromptLayerValidationError(PromptLayerError):
    """Exception raised when input validation fails.

    This can be due to invalid types, out of range values, or malformed data.
    """

    pass
