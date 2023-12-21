import logging
from typing import Dict, List, Optional, Union

from pydantic import StrictStr

from .constants import VisionAIErrorCode
from .error_messages import VAI_ERROR_MESSAGES_MAP

logger = logging.getLogger(__name__)


class VisionAIException(Exception):
    error_code: Optional[VisionAIErrorCode] = None
    error_message: Optional[str] = None
    message_kwargs: dict = {}

    def __init__(
        self, error_code: StrictStr, message_kwargs: Optional[dict] = None
    ) -> Dict[StrictStr, Union[StrictStr, List[StrictStr]]]:
        if message_kwargs is None:
            message_kwargs = dict()

        # We retrieve error message map for its error code
        error_message_str: dict[
            StrictStr, Union[List[StrictStr], StrictStr]
        ] = VAI_ERROR_MESSAGES_MAP[error_code]

        # we can assign message kwargs for these keys
        # since each keys is a string with variable
        new_error_message = ""
        try:
            new_error_message: StrictStr = error_message_str.format(**message_kwargs)
        except KeyError:
            logger.exception(f"Missing required string keys for {error_code}")

        self.error_code = error_code
        self.error_message = new_error_message
        self.message_kwargs = message_kwargs
        super().__init__(new_error_message)
