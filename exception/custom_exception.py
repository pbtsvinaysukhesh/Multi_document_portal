import sys
import traceback
from typing import Optional, cast
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentPortalException(Exception):
    def __init__(self, error_message, error_details: Optional[object] = None):
        # Normalize message
        if isinstance(error_message, BaseException):
            norm_msg = str(error_message)
        else:
            norm_msg = str(error_message)

        # Resolve exc_info (supports: sys module, Exception object, or current context)
        exc_type = exc_value = exc_tb = None
        if error_details is None:
            exc_type, exc_value, exc_tb = sys.exc_info()
        else:
            if hasattr(error_details, "exc_info"):  # e.g., sys
                #exc_type, exc_value, exc_tb = error_details.exc_info()
                exc_info_obj = cast(sys, error_details)
                exc_type, exc_value, exc_tb = exc_info_obj.exc_info()
            elif isinstance(error_details, BaseException):
                exc_type, exc_value, exc_tb = type(error_details), error_details, error_details.__traceback__
            else:
                exc_type, exc_value, exc_tb = sys.exc_info()

        # Walk to the last frame to report the most relevant location
        last_tb = exc_tb
        while last_tb and last_tb.tb_next:
            last_tb = last_tb.tb_next

        self.file_name = last_tb.tb_frame.f_code.co_filename if last_tb else "<unknown>"
        self.lineno = last_tb.tb_lineno if last_tb else -1
        self.error_message = norm_msg

        # Full pretty traceback (if available)
        if exc_type and exc_tb:
            self.traceback_str = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        else:
            self.traceback_str = ""

        super().__init__(self.__str__())

    def __str__(self):
        # Compact, logger-friendly message (no leading spaces)
        base = f"Error in [{self.file_name}] at line [{self.lineno}] | Message: {self.error_message}"
        if self.traceback_str:
            return f"{base}\nTraceback:\n{self.traceback_str}"
        return base

    def __repr__(self):
        return f"DocumentPortalException(file={self.file_name!r}, line={self.lineno}, message={self.error_message!r})"


class DocumentProcessingException(DocumentPortalException):
    """Exception for document processing errors."""
    pass


class UnsupportedFormatException(DocumentPortalException):
    """Exception for unsupported file formats."""
    pass


class FileNotFoundException(DocumentPortalException):
    """Exception for file not found errors."""
    pass


class PDFProcessingException(DocumentProcessingException):
    """Exception for PDF processing errors."""
    pass


class DOCXProcessingException(DocumentProcessingException):
    """Exception for DOCX processing errors."""
    pass


class ExcelProcessingException(DocumentProcessingException):
    """Exception for Excel processing errors."""
    pass


class CSVProcessingException(DocumentProcessingException):
    """Exception for CSV processing errors."""
    pass


class PowerPointProcessingException(DocumentProcessingException):
    """Exception for PowerPoint processing errors."""
    pass


class TextProcessingException(DocumentProcessingException):
    """Exception for text file processing errors."""
    pass


class MarkdownProcessingException(DocumentProcessingException):
    """Exception for Markdown processing errors."""
    pass


class SQLiteProcessingException(DocumentProcessingException):
    """Exception for SQLite database processing errors."""
    pass


# if __name__ == "__main__":
#     # Demo-1: generic exception -> wrap
#     try:
#         a = 1 / 0
#     except Exception as e:
#         raise DocumentPortalException("Division failed", e) from e

#     # Demo-2: still supports sys (old pattern)
#     # try:
#     #     a = int("abc")
#     # except Exception as e:
#     #     raise DocumentPortalException(e, sys)
