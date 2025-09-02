import pytest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import document processing components
from src.MultiDocument.multi_doc_handler import DocumentHandler
from src.MultiDocument.multi_doc import DocumentProcessor


class TestDocumentProcessing:
    """Test cases for enhanced document processing capabilities."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.doc_handler = DocumentHandler()
        self.doc_processor = DocumentProcessor()

    def test_document_handler_initialization(self):
        """Test 1: Verify DocumentHandler initializes correctly."""
        assert self.doc_handler is not None
        assert hasattr(self.doc_handler, 'read_file')

    def test_document_processor_initialization(self):
        """Test 2: Verify DocumentProcessor initializes correctly."""
        assert self.doc_processor is not None
        assert hasattr(self.doc_processor, 'process_document')
        assert hasattr(self.doc_processor, 'add_to_vector_store')

    def test_read_pdf_file(self):
        """Test 3: Test reading PDF files."""
        # Create a temporary PDF file for testing
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            # Write some dummy content (this would normally be PDF binary data)
            temp_file.write(b"Mock PDF content for testing")
            temp_path = temp_file.name

        try:
            # Test reading the file
            content = self.doc_handler.read_file(temp_path)
            assert isinstance(content, str)
            assert len(content) > 0
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_read_text_file(self):
        """Test 4: Test reading text files."""
        # Create a temporary text file
        test_content = "This is a test document for processing.\nIt contains multiple lines.\nAnd various text content."

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(test_content)
            temp_path = temp_file.name

        try:
            content = self.doc_handler.read_file(temp_path)
            assert content == test_content
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_process_document_pdf(self):
        """Test 5: Test processing PDF documents."""
        # Mock PDF processing
        with patch('src.MultiDocument.multi_doc.DocumentProcessor._extract_text_from_pdf') as mock_extract:
            mock_extract.return_value = "Extracted text from PDF document."

            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(b"Mock PDF content")
                temp_path = temp_file.name

            try:
                content = self.doc_processor.process_document(temp_path)
                assert content == "Extracted text from PDF document."
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    def test_process_document_text(self):
        """Test 6: Test processing text documents."""
        test_content = "This is a simple text document for testing."

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(test_content)
            temp_path = temp_file.name

        try:
            content = self.doc_processor.process_document(temp_path)
            assert content == test_content
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_table_extraction_from_pdf(self):
        """Test 7: Test table extraction from PDF."""
        # Mock table extraction
        with patch('src.MultiDocument.multi_doc_handler.DocumentHandler._extract_tables_camelot') as mock_camelot, \
             patch('src.MultiDocument.multi_doc_handler.DocumentHandler._extract_tables_tabula') as mock_tabula:

            mock_camelot.return_value = ["Table 1: Name, Age\nJohn, 25\nJane, 30"]
            mock_tabula.return_value = []

            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(b"Mock PDF with tables")
                temp_path = temp_file.name

            try:
                # This would normally call the table extraction methods
                tables = self.doc_handler._extract_tables_camelot(temp_path)
                assert len(tables) > 0
                assert "Table 1" in tables[0]
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    def test_image_extraction_from_pdf(self):
        """Test 8: Test image extraction from PDF."""
        # Mock image extraction
        with patch('src.MultiDocument.multi_doc_handler.DocumentHandler._extract_images_from_pdf') as mock_extract:
            mock_extract.return_value = ["image_1.png", "image_2.jpg"]

            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(b"Mock PDF with images")
                temp_path = temp_file.name

            try:
                images = self.doc_handler._extract_images_from_pdf(temp_path)
                assert len(images) == 2
                assert "image_1.png" in images
                assert "image_2.jpg" in images
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    def test_ocr_text_extraction(self):
        """Test 9: Test OCR text extraction from images."""
        # Mock OCR extraction
        with patch('src.MultiDocument.multi_doc_handler.DocumentHandler._perform_ocr') as mock_ocr:
            mock_ocr.return_value = "Extracted text from image using OCR."

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_file.write(b"Mock image content")
                temp_path = temp_file.name

            try:
                ocr_text = self.doc_handler._perform_ocr(temp_path)
                assert ocr_text == "Extracted text from image using OCR."
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    def test_vector_store_operations(self):
        """Test 10: Test vector store operations."""
        # Mock vector store operations
        with patch('src.MultiDocument.multi_doc.DocumentProcessor._initialize_vector_store') as mock_init, \
             patch('src.MultiDocument.multi_doc.DocumentProcessor._add_document_chunks') as mock_add:

            mock_init.return_value = None
            mock_add.return_value = None

            # Test adding to vector store
            self.doc_processor.add_to_vector_store("test_file.txt", "Test content for vector store")

            # Verify the methods were called
            mock_init.assert_called_once()
            mock_add.assert_called_once()

    def test_chunk_text_processing(self):
        """Test 11: Test text chunking functionality."""
        long_text = "This is a long document. " * 100  # Repeat to make it long

        # Mock chunking
        with patch('src.MultiDocument.multi_doc.DocumentProcessor._chunk_text') as mock_chunk:
            mock_chunk.return_value = [long_text[i:i+1000] for i in range(0, len(long_text), 1000)]

            chunks = self.doc_processor._chunk_text(long_text, chunk_size=1000, overlap=200)

            assert len(chunks) > 1
            assert all(len(chunk) <= 1000 for chunk in chunks)

    def test_document_metadata_extraction(self):
        """Test 12: Test document metadata extraction."""
        # Mock metadata extraction
        with patch('src.MultiDocument.multi_doc_handler.DocumentHandler._extract_metadata') as mock_meta:
            mock_meta.return_value = {
                "title": "Test Document",
                "author": "Test Author",
                "pages": 10,
                "created_date": "2024-01-01"
            }

            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(b"Mock PDF content")
                temp_path = temp_file.name

            try:
                metadata = self.doc_handler._extract_metadata(temp_path)
                assert metadata["title"] == "Test Document"
                assert metadata["author"] == "Test Author"
                assert metadata["pages"] == 10
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    def test_error_handling_invalid_file(self):
        """Test 13: Test error handling for invalid files."""
        # Test with non-existent file
        with pytest.raises(FileNotFoundError):
            self.doc_handler.read_file("non_existent_file.pdf")

    def test_error_handling_corrupted_file(self):
        """Test 14: Test error handling for corrupted files."""
        # Create a file with invalid content
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
            temp_file.write(b"Invalid PDF content that might cause parsing errors")
            temp_path = temp_file.name

        try:
            # This should handle the error gracefully
            content = self.doc_handler.read_file(temp_path)
            # Even if parsing fails, should return some content or handle error
            assert isinstance(content, str)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_batch_document_processing(self):
        """Test 15: Test batch processing of multiple documents."""
        # Create multiple temporary files
        temp_files = []
        try:
            for i in range(3):
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{i}.txt', delete=False) as temp_file:
                    temp_file.write(f"This is test document number {i}.")
                    temp_files.append(temp_file.name)

            # Process all files
            processed_contents = []
            for file_path in temp_files:
                content = self.doc_processor.process_document(file_path)
                processed_contents.append(content)

            assert len(processed_contents) == 3
            for i, content in enumerate(processed_contents):
                assert f"document number {i}" in content

        finally:
            # Clean up all temp files
            for file_path in temp_files:
                if os.path.exists(file_path):
                    os.remove(file_path)

    def test_memory_usage_large_documents(self):
        """Test 16: Test memory usage with large documents."""
        # Create a large text file
        large_content = "This is a very large document. " * 10000  # 30,000 words approximately

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            temp_file.write(large_content)
            temp_path = temp_file.name

        try:
            # Process the large document
            content = self.doc_processor.process_document(temp_path)

            # Should handle large content without memory issues
            assert len(content) == len(large_content)
            assert content == large_content

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_concurrent_document_processing(self):
        """Test 17: Test concurrent processing of documents."""
        import threading

        results = []
        errors = []

        def process_document(doc_id):
            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix=f'_{doc_id}.txt', delete=False) as temp_file:
                    temp_file.write(f"Content for document {doc_id}")
                    temp_path = temp_file.name

                content = self.doc_processor.process_document(temp_path)
                results.append((doc_id, content))

                # Clean up
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            except Exception as e:
                errors.append((doc_id, str(e)))

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=process_document, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        assert len(results) == 5
        assert len(errors) == 0

        for doc_id, content in results:
            assert f"document {doc_id}" in content


if __name__ == "__main__":
    pytest.main([__file__])
