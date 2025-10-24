import os
import numpy as np
import pandas as pd
import opensmile
from tqdm import tqdm
import glob
import soundfile as sf

class EGeMAPSExtractor:
    def __init__(self, feature_level="Functionals", debug: bool = False):
        """
        Initialize OpenSMILE for eGeMAPS feature extraction.
        Args:
            feature_level: 'Functionals' (default) or 'LowLevelDescriptors'
        """
        # Map to OpenSMILE enums
        level_map = {
            "Functionals": opensmile.FeatureLevel.Functionals,
            "LLDs": opensmile.FeatureLevel.LowLevelDescriptors
        }

        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=level_map.get(feature_level, opensmile.FeatureLevel.Functionals)
        )
        self.debug = debug

    def extract_from_file(self, file_path):
        """
        Extract eGeMAPS features from a single audio file.
        Returns: 1D numpy array of features
        """
        feat_df = self.smile.process_file(file_path)
        feat_vector = feat_df.values.flatten()
        if self.debug:
            print(f"[eGeMAPS] {os.path.basename(file_path)} -> shape={feat_vector.shape}, mean={feat_vector.mean():.4f}, std={feat_vector.std():.4f}")
        return feat_vector

    def extract_from_dir(self, data_dir, save_dir=None):
        """
        Batch extract features from all .wav files in a directory.
        Automatically labels samples using parent folder names.
        Args:
            data_dir: directory containing preprocessed audio (e.g., 'pitt_corpus_processed')
            save_dir: where to save .npy and .csv feature files (optional)
        Returns:
            features_np, labels_np
        """
        audio_files = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)
        features_list, labels_list = [], []

        print("🚀 Extracting OpenSMILE eGeMAPS features...")

        for file_path in tqdm(audio_files):
            try:
                feat_vector = self.extract_from_file(file_path)
                features_list.append(feat_vector)

                # Parent folder (class label)
                label = os.path.basename(os.path.dirname(file_path))
                labels_list.append(label)
            except Exception as e:
                print(f"⚠️ Error processing {file_path}: {e}")

        features_np = np.array(features_list)
        labels_np = np.array(labels_list)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            np.save(os.path.join(save_dir, "egemaps_features.npy"), features_np)
            np.save(os.path.join(save_dir, "egemaps_labels.npy"), labels_np)

            df = pd.DataFrame(features_np)
            df["label"] = labels_np
            df.to_csv(os.path.join(save_dir, "egemaps_features.csv"), index=False)

            print(f"✅ Features saved at: {save_dir}")
            print(f"Shape: {features_np.shape}")

        return features_np, labels_np

    def extract_from_segments(self, segments, sr=16000, tmp_path="temp_segment.wav"):
        """
        Extract features directly from audio segments in memory (for inference).
        Each segment is a numpy waveform array.
        Returns: numpy array of shape (num_segments, num_features)
        """
        features = []
        for i, seg in enumerate(segments):
            try:
                # Save temporarily (OpenSMILE works with files)
                tmp_file = f"{i}_{os.path.basename(tmp_path)}"
                sf.write(tmp_file, seg, sr)

                feat = self.extract_from_file(tmp_file)
                features.append(feat)

                os.remove(tmp_file)
            except Exception as e:
                print(f"⚠️ Error processing segment {i}: {e}")
        feats = np.array(features)
        print(f"[eGeMAPS] All segments -> shape={feats.shape}")
        return feats
