"""
Integration tests for spotlighting with DSPy modules.

These tests verify that the Spotlighting wrapper works correctly with
actual DSPy modules and LLM interactions.
"""

import pytest
import dspy
from spotlighting import Spotlighting
from unittest.mock import Mock, patch, MagicMock


class TestIntegrationWithDSpyPredict:
    """Integration tests with dspy.Predict modules."""

    def test_spotlighting_wraps_predict_module(self):
        """Test that Spotlighting can wrap a Predict module."""

        class Summarize(dspy.Signature):
            """Summarize a document."""
            document: str = dspy.InputField()
            summary: str = dspy.OutputField()

        predictor = dspy.Predict(Summarize)
        spotlighted = Spotlighting(predictor, method="datamarking", marker="^")

        assert spotlighted.module == predictor
        assert spotlighted.method == "datamarking"

    def test_forward_transforms_input_fields(self):
        """Test that forward() transforms input fields correctly."""

        class Summarize(dspy.Signature):
            """Summarize a document."""
            document: str = dspy.InputField()
            summary: str = dspy.OutputField()

        predictor = dspy.Predict(Summarize)
        spotlighted = Spotlighting(predictor, method="datamarking", marker="^")

        # The forward method should transform the document
        # We'll verify by checking what gets passed through
        transformed_input = None

        def capture_forward(**kwargs):
            nonlocal transformed_input
            transformed_input = kwargs
            # Return a mock prediction
            return dspy.Prediction(summary="Mocked summary")

        with patch.object(predictor, 'forward', side_effect=capture_forward):
            result = spotlighted(document="Hello world test")

            # Check that the input was transformed
            assert transformed_input is not None
            assert "document" in transformed_input
            assert transformed_input["document"] == "Hello^world^test"

    def test_multiple_input_fields(self):
        """Test spotlighting with multiple input fields."""

        class Translate(dspy.Signature):
            """Translate text."""
            source_text: str = dspy.InputField()
            target_language: str = dspy.InputField()
            translation: str = dspy.OutputField()

        predictor = dspy.Predict(Translate)
        spotlighted = Spotlighting(
            predictor,
            method="datamarking",
            fields=["source_text"],  # Only spotlight source_text
            marker="^"
        )

        transformed_input = None

        def capture_forward(**kwargs):
            nonlocal transformed_input
            transformed_input = kwargs
            return dspy.Prediction(translation="Mocked translation")

        with patch.object(predictor, 'forward', side_effect=capture_forward):
            result = spotlighted(
                source_text="Hello world",
                target_language="Spanish"
            )

            # Only source_text should be transformed
            assert transformed_input["source_text"] == "Hello^world"
            assert transformed_input["target_language"] == "Spanish"

    def test_encoding_transformation_integration(self):
        """Test encoding transformation in an integration scenario."""

        class Summarize(dspy.Signature):
            """Summarize a document."""
            document: str = dspy.InputField()
            summary: str = dspy.OutputField()

        predictor = dspy.Predict(Summarize)
        spotlighted = Spotlighting(predictor, method="encoding", encoding="base64")

        transformed_input = None

        def capture_forward(**kwargs):
            nonlocal transformed_input
            transformed_input = kwargs
            return dspy.Prediction(summary="Mocked summary")

        with patch.object(predictor, 'forward', side_effect=capture_forward):
            result = spotlighted(document="Test document")

            # The document should be base64 encoded
            import base64
            decoded = base64.b64decode(transformed_input["document"]).decode('utf-8')
            assert decoded == "Test document"

    def test_delimiting_transformation_integration(self):
        """Test delimiting transformation in an integration scenario."""

        class Summarize(dspy.Signature):
            """Summarize a document."""
            document: str = dspy.InputField()
            summary: str = dspy.OutputField()

        predictor = dspy.Predict(Summarize)
        spotlighted = Spotlighting(
            predictor,
            method="delimiting",
            delimiter="[START]",
            end_delimiter="[END]"
        )

        transformed_input = None

        def capture_forward(**kwargs):
            nonlocal transformed_input
            transformed_input = kwargs
            return dspy.Prediction(summary="Mocked summary")

        with patch.object(predictor, 'forward', side_effect=capture_forward):
            result = spotlighted(document="Test document")

            # The document should be delimited
            assert transformed_input["document"] == "[START]Test document[END]"


class TestIntegrationWithChainOfThought:
    """Integration tests with dspy.ChainOfThought modules."""

    def test_spotlighting_wraps_cot_module(self):
        """Test that Spotlighting can wrap a ChainOfThought module."""

        class Analyze(dspy.Signature):
            """Analyze a document."""
            document: str = dspy.InputField()
            analysis: str = dspy.OutputField()

        cot = dspy.ChainOfThought(Analyze)
        spotlighted = Spotlighting(cot, method="datamarking", marker="^")

        assert spotlighted.module == cot
        assert spotlighted.method == "datamarking"

    def test_cot_with_datamarking(self):
        """Test ChainOfThought with datamarking transformation."""

        class Analyze(dspy.Signature):
            """Analyze a document."""
            document: str = dspy.InputField()
            analysis: str = dspy.OutputField()

        cot = dspy.ChainOfThought(Analyze)
        spotlighted = Spotlighting(cot, method="datamarking", marker="@")

        transformed_input = None

        def capture_forward(**kwargs):
            nonlocal transformed_input
            transformed_input = kwargs
            return dspy.Prediction(
                analysis="Mocked analysis",
                rationale="Mocked rationale"
            )

        with patch.object(cot, 'forward', side_effect=capture_forward):
            result = spotlighted(document="Test document here")

            # The document should be datamarked
            assert transformed_input["document"] == "Test@document@here"


class TestNonStringFields:
    """Test that non-string fields are not transformed."""

    def test_integer_field_not_transformed(self):
        """Test that integer fields are not transformed."""

        class Score(dspy.Signature):
            """Score a document."""
            document: str = dspy.InputField()
            threshold: int = dspy.InputField()
            score: int = dspy.OutputField()

        predictor = dspy.Predict(Score)
        spotlighted = Spotlighting(predictor, method="datamarking", marker="^")

        transformed_input = None

        def capture_forward(**kwargs):
            nonlocal transformed_input
            transformed_input = kwargs
            return dspy.Prediction(score=85)

        with patch.object(predictor, 'forward', side_effect=capture_forward):
            result = spotlighted(document="Test document", threshold=80)

            # Document should be transformed, threshold should not
            assert transformed_input["document"] == "Test^document"
            assert transformed_input["threshold"] == 80


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_document(self):
        """Test spotlighting with empty document."""

        class Summarize(dspy.Signature):
            """Summarize a document."""
            document: str = dspy.InputField()
            summary: str = dspy.OutputField()

        predictor = dspy.Predict(Summarize)
        spotlighted = Spotlighting(predictor, method="datamarking", marker="^")

        transformed_input = None

        def capture_forward(**kwargs):
            nonlocal transformed_input
            transformed_input = kwargs
            return dspy.Prediction(summary="Empty document")

        with patch.object(predictor, 'forward', side_effect=capture_forward):
            result = spotlighted(document="")

            # Empty string should remain empty
            assert transformed_input["document"] == ""

    def test_unicode_content(self):
        """Test spotlighting with Unicode content."""

        class Summarize(dspy.Signature):
            """Summarize a document."""
            document: str = dspy.InputField()
            summary: str = dspy.OutputField()

        predictor = dspy.Predict(Summarize)
        spotlighted = Spotlighting(predictor, method="datamarking", marker="^")

        transformed_input = None

        def capture_forward(**kwargs):
            nonlocal transformed_input
            transformed_input = kwargs
            return dspy.Prediction(summary="Mocked summary")

        with patch.object(predictor, 'forward', side_effect=capture_forward):
            unicode_text = "Hello 世界 test émojis 🎉"
            result = spotlighted(document=unicode_text)

            # Should handle Unicode correctly
            expected = "Hello^世界^test^émojis^🎉"
            assert transformed_input["document"] == expected

    def test_very_long_document(self):
        """Test spotlighting with a very long document."""

        class Summarize(dspy.Signature):
            """Summarize a document."""
            document: str = dspy.InputField()
            summary: str = dspy.OutputField()

        predictor = dspy.Predict(Summarize)
        spotlighted = Spotlighting(predictor, method="datamarking", marker="^")

        transformed_input = None

        def capture_forward(**kwargs):
            nonlocal transformed_input
            transformed_input = kwargs
            return dspy.Prediction(summary="Mocked summary")

        with patch.object(predictor, 'forward', side_effect=capture_forward):
            # Create a long document
            long_doc = " ".join(["word"] * 10000)
            result = spotlighted(document=long_doc)

            # Should handle long documents
            assert "^" in transformed_input["document"]
            assert transformed_input["document"].count("^") == 9999  # 10000 words - 1
