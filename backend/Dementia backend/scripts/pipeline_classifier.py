from scripts.audio_processor import AudioPreprocessor
from scripts.egemaps_extractor import EGeMAPSExtractor
from scripts.dementia_classifier import DementiaClassifier

class DementiaPipeline:
    def __init__(self, model_path="models/CNN_final.pth", debug: bool = False):
        """
        Initialize all components:
        - Audio preprocessing
        - eGeMAPS feature extraction
        - CNN classifier
        """
        self.preprocessor = AudioPreprocessor(debug=debug)
        self.feature_extractor = EGeMAPSExtractor(debug=debug)
        self.classifier = DementiaClassifier(model_path=model_path, debug=debug)
        self.debug = debug

    def classify(self, audio_path):
        """
        Full pipeline:
        1. Preprocess audio
        2. Extract eGeMAPS features
        3. Classify each segment and aggregate
        Returns:
            dict: {'confidence': float, 'category': str}
        """
        # Step 1: Preprocess audio and get segments
        segments = self.preprocessor.preprocess(audio_path)
        
        print(f"[Pipeline] Segments: count={len(segments)}, each ~{self.preprocessor.segment_duration}s @ {self.preprocessor.target_sr}Hz")

        # Step 2: Extract features for all segments
        features = self.feature_extractor.extract_from_segments(segments)
        
        print(f"[Pipeline] Features shape: {features.shape}")

        # Step 3: Classify and aggregate
        result = self.classifier.classify(features)
        
        print(f"[Pipeline] Final result: {result}")

        return result
