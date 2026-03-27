import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp

mp_hands = mp.solutions.hands


class HandEmbedder(nn.Module):
    """
    Extracts MediaPipe hand landmarks from batched frame tensors.
    """

    def __init__(self, num_hands = 2):
        super().__init__()
        self.num_hands = num_hands
        self.hand_emb_dim = num_hands * 63
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: (N, C, H, W) float tensor

        Returns:
            (N, num_hands * 63) float32 tensor on the same device as frames.
        """
        device = frames.device

        # Convert to uint8 numpy (H, W, C) for MediaPipe
        frames_np = frames.detach().cpu()
        if frames_np.max() <= 1.0:
            frames_np = (frames_np * 255.0).clamp(0, 255)
        frames_np = frames_np.to(torch.uint8).permute(0, 2, 3, 1).numpy()

        batch_embeddings = []
        for frame in frames_np:
            results = self.hands.process(frame)

            embeddings = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    embedding = []
                    for lm in hand_landmarks.landmark:
                        embedding.extend([lm.x, lm.y, lm.z])
                    embeddings.append(np.array(embedding))

            while len(embeddings) < self.num_hands:
                embeddings.append(np.zeros(63, dtype=np.float32))

            batch_embeddings.append(np.concatenate(embeddings[:self.num_hands]))

        out = np.stack(batch_embeddings, axis=0)  # (N, num_hands * 63)
        return torch.from_numpy(out).to(device=device, dtype=torch.float32)
