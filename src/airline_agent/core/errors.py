class AirlineBusinessError(Exception):
    """航空工具执行时可预期的业务错误。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message