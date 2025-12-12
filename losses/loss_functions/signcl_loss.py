# losses/loss_functions/signcl_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class SignCL(nn.Module):
    def __init__(self, max_distance=64.0, pos_samples=2, neg_samples=4):
        """
        SignCL contrastive loss for frame-level embeddings.
        """
        super(SignCL, self).__init__()
        self.max_distance = max_distance
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

    def forward(self, inputs_embeds, margin=20):
        """
        Compute SignCL loss.

        Args:
            inputs_embeds: (batch_size, seq_len, embed_dim)
            margin: minimum margin for negative sampling
            
        """
     
        batch_size, seq_len, _ = inputs_embeds.size()
        total_loss = 0

        for i in range(1, seq_len - 2):
            anchor = inputs_embeds[:, i, :].unsqueeze(1)

            # Positive samples
            pos_indices = [idx for idx in range(max(0, i - 1), min(seq_len, i + 2)) if idx != i]
            selected_pos_indices = random.sample(pos_indices, k=min(len(pos_indices), self.pos_samples))
            positives = inputs_embeds[:, selected_pos_indices, :]

            # Negative samples
            neg_indices = [idx for idx in range(2, seq_len - 2) if idx < i - margin or idx > i + margin]
            selected_neg_indices = random.sample(neg_indices, k=min(len(neg_indices), self.neg_samples))
            negatives = inputs_embeds[:, selected_neg_indices, :]

            # Distances
            pos_dist = torch.sum(torch.abs(anchor - positives), dim=-1)
            neg_dist = torch.sum(torch.abs(anchor - negatives), dim=-1)
            # pos_dist = torch.mean(torch.abs(anchor - positives), dim=-1)
            # neg_dist = torch.mean(torch.abs(anchor - negatives), dim=-1)

            # Loss
            pos_loss = F.softplus(pos_dist - self.max_distance).mean()
            neg_loss = F.softplus(self.max_distance - neg_dist).mean()
            total_loss += (pos_loss + neg_loss)

        # total_loss /= (batch_size * (seq_len - 4))
        total_loss /= (batch_size * (seq_len - 3))
        return total_loss

# Adapter for Sign2GPT base_loss
class Loss(nn.Module):
    def __init__(self, max_distance=64.0, pos_samples=2, neg_samples=4): #chnaging max_distance=32.0 to 1.0 on using mean() for distance calculation
        super().__init__()
        self.signcl = SignCL(max_distance=max_distance, pos_samples=pos_samples, neg_samples=neg_samples)
        # self.signcl = SignCL(max_distance=1.0, pos_samples=pos_samples, neg_samples=neg_samples)

    def forward(self, y_pred, target):
        """
        Forward compatible with base_loss: y_pred and target dictionaries.
        """
        # print('y_pred', y_pred['enc_output']['list_of_features'][0][0])
        # frames_feature = y_pred['enc_output']['list_of_features'][0][0]
        frames_feature = y_pred['enc_output']['hidden_state']
        # Optional dynamic margin[0]
        num_frames = frames_feature.size(1)
        # print('num_frames', num_frames)
        # print('target', target)
        pseudo_gloss_ids_list = target["pseudo_gloss_ids"]

# Compute text_length for margin dynamically
        text_lengths = [t.shape[0] for t in pseudo_gloss_ids_list]
        text_length = max(text_lengths) 
        # margin = min(20, max(10, int(num_frames // text_length * 2.3)))
        # num_frames and text_length already computed
        # margin = max(10, int((num_frames // text_length + 1) * 2.3)) * 2
        
        # num_negative = 30  # or 4 × neg_samples
        # margin = min(margin, int((num_frames - num_negative) / 2))
        margin = max(20, int((num_frames // text_length + 1) * 2.3))
        
        return self.signcl(frames_feature, margin=margin)
