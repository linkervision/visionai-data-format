import logging
from typing import Dict, List, Optional, Union

from pydantic import StrictStr

from .error_messages import VAI_ERROR_MESSAGES_MAP

logger = logging.getLogger(__name__)


class VisionAIException(Exception):
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

        super().__init__(new_error_message)
