import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# MLP architecture matching the saved checkpoint (88 -> 512 -> 256 -> 2)
class CNNModel(nn.Module):
    def __init__(self, input_dim=88):
        super(CNNModel, self).__init__()
        # Keep layer names to match checkpoint keys
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)  # binary output: Control vs Dementia

    def forward(self, x):
        # x: (batch, features)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- Dementia Classifier ---
class DementiaClassifier:
    def __init__(
        self,
        model_path: str = "models/CNN_final.pth",
        input_dim: int = 88,
        scaler_path: str | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
        positive_class_index: int = 1,
        aggregate: str = "probs",  # "probs" or "logits"
        debug: bool = False,
    ):
        """
        Args:
            model_path: path to trained .pth file
            input_dim: number of features expected by model (default 88 for eGeMAPS)
            scaler_path: optional path to npz (keys: mean, std) or npy (tuple/list) with training normalization stats
            feature_mean/std: provide mean and std arrays directly instead of scaler_path
            positive_class_index: which index corresponds to 'Dementia' in the model output
            aggregate: averaging method across segments ("probs" or "logits")
            debug: print detailed debug info
        """
        self.device = torch.device("cpu")
        self.model = CNNModel(input_dim=input_dim)
        # Load checkpoint strictly now that architecture matches
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()  # inference mode

        self.debug = debug
        self.positive_class_index = positive_class_index
        self.aggregate = aggregate

        # Load normalization stats if provided
        self.feature_mean = None
        self.feature_std = None
        if scaler_path is not None:
            try:
                if scaler_path.endswith(".npz"):
                    npz = np.load(scaler_path)
                    self.feature_mean = np.array(npz["mean"], dtype=np.float32)
                    self.feature_std = np.array(npz["std"], dtype=np.float32)
                else:
                    arr = np.load(scaler_path, allow_pickle=True)
                    if isinstance(arr, np.ndarray) and arr.dtype == object and len(arr) == 2:
                        self.feature_mean = np.array(arr[0], dtype=np.float32)
                        self.feature_std = np.array(arr[1], dtype=np.float32)
                if self.debug:
                    print(f"[Classifier] Loaded scaler from {scaler_path}: mean/std shapes = {None if self.feature_mean is None else self.feature_mean.shape}/{None if self.feature_std is None else self.feature_std.shape}")
            except Exception as e:
                print(f"[Classifier] Warning: failed to load scaler from {scaler_path}: {e}")
        if feature_mean is not None and feature_std is not None:
            self.feature_mean = np.array(feature_mean, dtype=np.float32)
            self.feature_std = np.array(feature_std, dtype=np.float32)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.feature_mean is None or self.feature_std is None:
            return x
        if x.shape[-1] != self.feature_mean.shape[-1]:
            print(
                f"[Classifier] Warning: feature dim {x.shape[-1]} != scaler dim {self.feature_mean.shape[-1]}; skipping normalization"
            )
            return x
        std = np.where(self.feature_std == 0, 1.0, self.feature_std)
        return (x - self.feature_mean) / std

    def _predict_segment(self, features):
        """
        Predict confidence for a single feature vector.
        Args:
            features: 1D numpy array
        Returns:
            confidence: float [0,1] for Dementia
        """
        with torch.no_grad():
            x_np = features.astype(np.float32)
            if self.debug:
                print(f"[Classifier] Segment feature stats: shape={x_np.shape}, mean={x_np.mean():.4f}, std={x_np.std():.4f}, min={x_np.min():.4f}, max={x_np.max():.4f}")
            x_np = self._normalize(x_np)
            if self.debug:
                print(f"[Classifier] After normalization: mean={x_np.mean():.4f}, std={x_np.std():.4f}")

            x = torch.from_numpy(x_np).unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy().flatten()
            if self.debug:
                print(f"[Classifier] Logits: {logits.cpu().numpy().flatten()} | Probs: {probs}")
            confidence = float(probs[self.positive_class_index])  # Dementia class probability
        return confidence, logits.cpu().numpy().flatten(), probs

    def classify(self, features_array):
        """
        Classify audio with multiple segments.
        Args:
            features_array: numpy array (num_segments, num_features)
        Returns:
            dict with average confidence and category
        """
        if len(features_array.shape) == 1:
            features_array = features_array[np.newaxis, :]  # single segment

        # Shape/compatibility checks
        in_features = self.model.fc1.in_features
        if features_array.shape[1] != in_features:
            raise ValueError(
                f"Feature dimension mismatch: features have {features_array.shape[1]} dims but model expects {in_features}. "
                "Ensure the OpenSMILE feature set/level and preprocessing match training."
            )

        # Per-file normalization fallback if no scaler provided
        feats_proc = features_array.astype(np.float32)
        if self.feature_mean is None or self.feature_std is None:
            # Compute across all segments
            mean_f = np.mean(feats_proc, axis=0)
            std_f = np.std(feats_proc, axis=0)
            std_f = np.where(std_f == 0, 1.0, std_f)
            feats_proc = (feats_proc - mean_f) / std_f

            print("[Classifier] Using per-file normalization (no external scaler provided)")
        else:
            # Apply provided scaler
            feats_proc = np.vstack([self._normalize(f) for f in feats_proc])

        # Replace NaNs/Infs and optional clipping
        feats_proc = np.nan_to_num(feats_proc, nan=0.0, posinf=0.0, neginf=0.0)
        feats_proc = np.clip(feats_proc, -10.0, 10.0)

        confidences = []
        logits_list = []
        probs_list = []
        for i, seg_feat in enumerate(feats_proc):
            conf, logits, probs = self._predict_segment(seg_feat)
            confidences.append(conf)
            logits_list.append(logits)
            probs_list.append(probs)

            pred_idx = int(np.argmax(probs))
            print(f"[Classifier] Segment {i}: pred_class={pred_idx}, dementia_prob={probs[self.positive_class_index]:.6f}")

        if self.aggregate == "logits":
            avg_logits = np.mean(np.vstack(logits_list), axis=0)
            avg_probs = np.exp(avg_logits - np.max(avg_logits))
            avg_probs = avg_probs / np.sum(avg_probs)
            avg_conf = float(avg_probs[self.positive_class_index])
            if self.debug:
                print(f"[Classifier] Aggregation by logits -> avg logits: {avg_logits}, avg probs: {avg_probs}")
        else:
            avg_conf = float(np.mean(confidences))
            if self.debug:
                print(f"[Classifier] Aggregation by probs -> per-seg mean: {np.mean(probs_list, axis=0)}")

        # Map to 5-category label
        if avg_conf <= 0.2:
            category = "Definitely Not Dementia"
        elif avg_conf <= 0.4:
            category = "Likely Not Dementia"
        elif avg_conf <= 0.6:
            category = "Borderline / Uncertain"
        elif avg_conf <= 0.8:
            category = "Likely Dementia"
        else:
            category = "Definitely Dementia"

        return {
            "confidence": round(avg_conf, 4),
            "category": category
        }
