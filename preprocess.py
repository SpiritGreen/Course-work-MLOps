"""ClearML Serving preprocessing hook for the sentiment model."""
from typing import Any


class Preprocess:
    """Preprocessing and postprocessing hooks called by ClearML Serving per request."""

    def preprocess(self, body: dict, _state: dict, _collect_custom_statistics_fn=None) -> Any:
        """Extract text from request body and wrap in a list for sklearn Pipeline.predict()."""
        return [body.get("text", "")]

    def postprocess(self, data: Any, _state: dict, _collect_custom_statistics_fn=None) -> dict:
        """Wrap the model prediction array into a JSON-serialisable response dict."""
        return {"label": str(data[0])}
