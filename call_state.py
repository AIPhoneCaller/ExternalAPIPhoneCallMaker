# call_state.py
from enum import Enum, auto

class CallState(Enum):
    IDLE = auto()
    AI_SPEAKING = auto()
    USER_LISTENING = auto()
    USER_SPEAKING = auto()
    AI_THINKING = auto()
    END_CALL = auto()
