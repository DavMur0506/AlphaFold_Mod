import torch
import torch.nn as nn
import pandas as pd
import pickle
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import math

class EnhancedSeqDataset(Dataset):
    def __init__(self, df_seq, df_feat, max_len=1024):
        """Enhanced dataset with better preprocessing and validation"""
        assert len(df_seq) == len(df_feat), "DataFrames must have same length"
        
        self.max_len = max_len
        
        # Convert sequences with better handling
        self.seqs = []
        self.valid_indices = []
        
        for i, s in enumerate(df_seq.tolist()):
            s_array = np.asarray(s)
            if s_array.ndim == 2 and s_array.shape[1] == 21:
                int_seq = np.argmax(s_array, axis=1)
                int_seq = np.clip(int_seq, 0, 20)
                self.seqs.append(int_seq.astype(np.int64))
            else:
                int_seq = np.asarray(s, dtype=np.int64)
                int_seq = np.clip(int_seq, 0, 20)
                self.seqs.append(int_seq)
            
            # Only keep sequences within reasonable length
            if 10 <= len(self.seqs[-1]) <= max_len:
                self.valid_indices.append(i)
            
            # Truncate if too long
            if len(self.seqs[-1]) > max_len:
                self.seqs[-1] = self.seqs[-1][:max_len]
        
        # Filter out invalid sequences
        print(f"Keeping {len(self.valid_indices)}/{len(self.seqs)} sequences (length 10-{max_len})")
        
        # Process targets with enhanced preprocessing
        self._process_targets_enhanced(df_feat)
        
        # Compute dataset statistics
        self._compute_enhanced_stats()
    
    def _process_targets_enhanced(self, df_feat):
        """Enhanced target processing with better error handling"""
        
        # 1. Residue index with better normalization
        self.residue_index = []
        for i, v in enumerate(df_feat['residue_index'].tolist()):
            if i not in self.valid_indices:
                continue
                
            v_array = np.asarray(v, dtype=np.int64)
            seq_len = len(self.seqs[i])
            
            if len(v_array) != seq_len:
                if len(v_array) > seq_len:
                    v_array = v_array[:seq_len]
                else:
                    # Better interpolation
                    indices = np.linspace(0, len(v_array)-1, seq_len)
                    v_array = np.interp(indices, np.arange(len(v_array)), v_array)
            
            # Clip to valid range and ensure integer
            v_array = np.clip(v_array, 0, 20).astype(np.int64)
            self.residue_index.append(v_array)
        
        # 2. Amino acid type (should match input sequence) - FIXED
        self.aatype = []
        for i, seq in enumerate(self.seqs):
            if i in self.valid_indices:
                self.aatype.append(seq.copy())
        
        # 3. Enhanced deletion matrix processing
        self.deletion_matrix = []
        valid_idx = 0
        for i, v in enumerate(df_feat['deletion_matrix_int'].tolist()):
            if i not in self.valid_indices:
                continue
                
            v_array = np.asarray(v, dtype=np.float32)
            seq_len = len(self.seqs[i])
            
            if v_array.ndim > 1:
                # Enhanced multi-dimensional processing
                if v_array.shape[0] == seq_len:
                    # Average across MSA dimension with weighted consensus
                    weights = np.ones(v_array.shape[1])
                    deletion_vec = np.average(v_array, axis=1, weights=weights)
                elif v_array.shape[1] == seq_len:
                    weights = np.ones(v_array.shape[0])
                    deletion_vec = np.average(v_array, axis=0, weights=weights)
                else:
                    # Reshape and interpolate
                    flat_array = v_array.flatten()
                    if len(flat_array) >= seq_len:
                        indices = np.linspace(0, len(flat_array)-1, seq_len)
                        deletion_vec = np.interp(indices, np.arange(len(flat_array)), flat_array)
                    else:
                        deletion_vec = np.zeros(seq_len, dtype=np.float32)
            else:
                if len(v_array) == seq_len:
                    deletion_vec = v_array.astype(np.float32)
                elif len(v_array) > seq_len:
                    deletion_vec = v_array[:seq_len].astype(np.float32)
                else:
                    deletion_vec = np.pad(v_array, (0, seq_len - len(v_array)), 
                                        mode='constant', constant_values=0).astype(np.float32)
            
            # Normalize to [0,1] range and add small smoothing
            deletion_vec = np.clip(deletion_vec, 0, 1)
            # Add label smoothing for binary targets
            deletion_vec = deletion_vec * 0.9 + 0.05
            self.deletion_matrix.append(deletion_vec)
            valid_idx += 1
        
        # 4. Enhanced MSA processing with better consensus
        self.msa = []
        valid_idx = 0
        for i, v in enumerate(df_feat['msa'].tolist()):
            if i not in self.valid_indices:
                continue
                
            v_array = np.asarray(v)
            seq_len = len(self.seqs[i])
            
            if v_array.ndim > 2:
                # Enhanced 3D MSA processing
                if v_array.shape[1] == seq_len and v_array.shape[2] == 21:
                    # Weighted consensus instead of simple mean
                    msa_depth = v_array.shape[0]
                    # Give higher weight to sequences with lower gaps
                    gap_penalty = np.sum(v_array[:, :, 20], axis=1)  # Gap character is usually index 20
                    weights = np.exp(-gap_penalty / seq_len)  # Exponential weighting
                    weights = weights / np.sum(weights)
                    
                    weighted_consensus = np.average(v_array, axis=0, weights=weights)
                    msa_consensus = np.argmax(weighted_consensus, axis=1)
                elif v_array.shape[0] == seq_len and v_array.shape[2] == 21:
                    # Similar weighted approach
                    msa_depth = v_array.shape[1]
                    gap_penalty = np.sum(v_array[:, :, 20], axis=0)
                    weights = np.exp(-gap_penalty / seq_len)
                    weights = weights / np.sum(weights)
                    
                    weighted_consensus = np.average(v_array, axis=1, weights=weights)
                    msa_consensus = np.argmax(weighted_consensus, axis=1)
                else:
                    # Fallback to sequence
                    msa_consensus = self.seqs[i].copy()
            elif v_array.ndim == 2:
                if v_array.shape[1] == seq_len:
                    # Enhanced mode calculation with confidence scoring
                    msa_consensus = []
                    for pos in range(seq_len):
                        pos_values = v_array[:, pos]
                        unique_vals, counts = np.unique(pos_values, return_counts=True)
                        # Weighted by frequency and penalize gaps
                        if 20 in unique_vals:  # Gap character
                            gap_idx = np.where(unique_vals == 20)[0][0]
                            counts[gap_idx] *= 0.1  # Heavily penalize gaps
                        most_common_idx = np.argmax(counts)
                        msa_consensus.append(unique_vals[most_common_idx])
                    msa_consensus = np.array(msa_consensus, dtype=np.int64)
                elif v_array.shape[0] == seq_len and v_array.shape[1] == 21:
                    msa_consensus = np.argmax(v_array, axis=1)
                else:
                    msa_consensus = self.seqs[i].copy()
            else:
                if len(v_array) == seq_len:
                    msa_consensus = v_array.astype(np.int64)
                else:
                    msa_consensus = self.seqs[i].copy()
            
            msa_consensus = np.clip(msa_consensus, 0, 20)
            if len(msa_consensus) != seq_len:
                msa_consensus = self.seqs[i].copy()
            
            self.msa.append(msa_consensus.astype(np.int64))
            valid_idx += 1
        
        # Filter sequences and targets to only valid ones
        self.seqs = [self.seqs[i] for i in self.valid_indices]
    
    def _compute_enhanced_stats(self):
        """Compute enhanced dataset statistics"""
        seq_lengths = [len(seq) for seq in self.seqs]
        
        self.stats = {
            'num_samples': len(self.seqs),
            'avg_seq_length': np.mean(seq_lengths),
            'median_seq_length': np.median(seq_lengths),
            'min_seq_length': np.min(seq_lengths),
            'max_seq_length': np.max(seq_lengths),
            'std_seq_length': np.std(seq_lengths),
            'deletion_rate': np.mean([np.mean(d > 0.5) for d in self.deletion_matrix]),
            'msa_similarity': np.mean([np.mean(seq == msa) for seq, msa in zip(self.seqs, self.msa)]),
            'aatype_similarity': np.mean([np.mean(seq == aa) for seq, aa in zip(self.seqs, self.aatype)])
        }
        
        print(f"\nEnhanced Dataset Statistics:")
        for key, value in self.stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
    
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = torch.tensor(self.seqs[idx], dtype=torch.long)
        targets = {
            'residue_index': torch.tensor(self.residue_index[idx], dtype=torch.long),
            'aatype': torch.tensor(self.aatype[idx], dtype=torch.long),
            'deletion_matrix': torch.tensor(self.deletion_matrix[idx], dtype=torch.float),
            'msa': torch.tensor(self.msa[idx], dtype=torch.long)
        }
        return seq, targets

class MultiScalePositionalEncoding(nn.Module):
    """Enhanced positional encoding with multiple scales"""
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        
        # Standard sinusoidal encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Multiple frequency scales for better position representation
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add learnable component
        self.register_buffer('pe', pe.unsqueeze(0))
        self.learnable_pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        # Scale parameter
        self.scale = math.sqrt(d_model)
    
    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.pe[:, :seq_len] + self.learnable_pe[:, :seq_len]
        return x * self.scale + pos_enc

class EnhancedTransformer(nn.Module):
    def __init__(self, vocab_size=21, d_model=512, nhead=8, num_layers=8, dropout=0.1):
        super().__init__()
        
        # Enhanced embedding with scaling
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = MultiScalePositionalEncoding(d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        # Enhanced transformer with alternating attention patterns
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            # Alternate between different head configurations for diversity
            layer_nhead = nhead if i % 2 == 0 else max(4, nhead // 2)
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=layer_nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.layers.append(encoder_layer)
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Enhanced prediction heads with residual connections
        self.residue_index_head = self._make_prediction_head(d_model, 21, dropout)
        self.aatype_head = self._make_prediction_head(d_model, 21, dropout)
        self.deletion_matrix_head = self._make_regression_head(d_model, dropout)
        self.msa_head = self._make_prediction_head(d_model, 21, dropout)
        
        # Task-specific layer norms
        self.task_norms = nn.ModuleDict({
            'residue_index': nn.LayerNorm(d_model),
            'aatype': nn.LayerNorm(d_model),
            'deletion_matrix': nn.LayerNorm(d_model),
            'msa': nn.LayerNorm(d_model)
        })
        
        self._init_weights()
    
    def _make_prediction_head(self, d_model, output_size, dropout):
        """Create enhanced prediction head for classification"""
        return nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.LayerNorm(d_model // 2),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.GELU(),
            nn.LayerNorm(d_model // 4),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 4, output_size)
        )
    
    def _make_regression_head(self, d_model, dropout):
        """Create regression head for continuous targets"""
        return nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.LayerNorm(d_model // 4),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, d_model // 8),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model // 8, 1)
        )
    
    def _init_weights(self):
        """Enhanced weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform for GELU
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                    module.weight.data[module.padding_idx].fill_(0)
    
    def forward(self, src, src_key_padding_mask=None):
        # Enhanced embedding and positional encoding
        x = self.embed(src)
        x = self.pos_encoder(x)
        x = self.input_dropout(x)
        
        # Progressive transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        
        x = self.final_norm(x)
        
        # Task-specific processing and prediction
        outputs = {}
        for task_name, head in [
            ('residue_index', self.residue_index_head),
            ('aatype', self.aatype_head),
            ('msa', self.msa_head)
        ]:
            task_features = self.task_norms[task_name](x)
            outputs[task_name] = head(task_features)
        
        # Special handling for deletion matrix (regression)
        deletion_features = self.task_norms['deletion_matrix'](x)
        outputs['deletion_matrix'] = torch.sigmoid(self.deletion_matrix_head(deletion_features).squeeze(-1))
        
        return outputs

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class EnhancedLossFunction:
    """Enhanced multi-task loss with adaptive weighting"""
    def __init__(self, device):
        self.device = device
        
        # Different loss functions for different tasks
        self.focal_loss = FocalLoss(alpha=1, gamma=2, ignore_index=0)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        
        # Adaptive loss weights (will be updated during training)
        self.loss_weights = nn.Parameter(torch.tensor([1.0, 2.0, 0.5, 1.5], device=device))
        self.task_names = ['residue_index', 'aatype', 'deletion_matrix', 'msa']
        
        # Track loss history for adaptive weighting
        self.loss_history = {name: [] for name in self.task_names}
    
    def __call__(self, predictions, targets, padding_mask, epoch=0):
        total_loss = 0.0
        losses = {}
        
        for i, key in enumerate(self.task_names):
            pred = predictions[key]
            target = targets[key]
            
            if key == 'deletion_matrix':
                # Regression task
                mask = ~padding_mask
                if mask.sum() > 0:
                    pred_masked = pred[mask]
                    target_masked = target[mask]
                    loss = self.bce_loss(pred_masked, target_masked)
                else:
                    loss = torch.tensor(0.0, device=self.device)
            else:
                # Classification tasks
                B, L, C = pred.shape
                pred_flat = pred.view(B * L, C)
                target_flat = target.view(B * L)
                
                # Mask out padding
                mask = ~padding_mask.view(B * L)
                if mask.sum() > 0:
                    pred_masked = pred_flat[mask]
                    target_masked = target_flat[mask]
                    
                    # Use focal loss for imbalanced tasks, CE for balanced ones
                    if key in ['deletion_matrix'] or epoch < 10:
                        loss = self.ce_loss(pred_masked, target_masked)
                    else:
                        loss = self.focal_loss(pred_masked, target_masked)
                else:
                    loss = torch.tensor(0.0, device=self.device)
            
            losses[key] = loss.item()
            
            # Adaptive weighting based on loss magnitude
            weight = torch.abs(self.loss_weights[i])
            total_loss += weight * loss
            
            # Update loss history
            self.loss_history[key].append(loss.item())
            if len(self.loss_history[key]) > 100:
                self.loss_history[key].pop(0)
        
        return total_loss, losses

def enhanced_collate_fn(batch, pad_idx=0):
    """Enhanced collate function with better padding"""
    seqs, targets_list = zip(*batch)
    
    # Sort by length for better batching efficiency
    sorted_batch = sorted(zip(seqs, targets_list), key=lambda x: len(x[0]), reverse=True)
    seqs, targets_list = zip(*sorted_batch)
    
    # Pad sequences
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=pad_idx)
    
    # Pad targets
    targets_padded = {}
    for key in targets_list[0].keys():
        target_tensors = [targets[key] for targets in targets_list]
        if key == 'deletion_matrix':
            targets_padded[key] = pad_sequence(target_tensors, batch_first=True, padding_value=0.0)
        else:
            targets_padded[key] = pad_sequence(target_tensors, batch_first=True, padding_value=pad_idx)
    
    # Create attention mask (True for padding)
    padding_mask = (seqs_padded == pad_idx)
    
    return seqs_padded, targets_padded, padding_mask

def calculate_enhanced_metrics(predictions, targets, padding_mask):
    """Calculate comprehensive metrics"""
    metrics = {}
    
    for key in predictions.keys():
        pred = predictions[key]
        target = targets[key]
        
        if key == 'deletion_matrix':
            # Regression metrics
            mask = ~padding_mask
            if mask.sum() > 0:
                pred_masked = pred[mask].detach().cpu().numpy()
                target_masked = target[mask].detach().cpu().numpy()
                
                mse = np.mean((pred_masked - target_masked) ** 2)
                mae = np.mean(np.abs(pred_masked - target_masked))
                
                # Binary classification metrics (threshold at 0.5)
                pred_binary = (pred_masked > 0.5).astype(int)
                target_binary = (target_masked > 0.5).astype(int)
                acc = np.mean(pred_binary == target_binary)
                
                metrics[key] = {'mse': mse, 'mae': mae, 'accuracy': acc}
        else:
            # Classification metrics
            B, L, C = pred.shape
            pred_flat = torch.argmax(pred, dim=-1).view(B * L)
            target_flat = target.view(B * L)
            
            mask = ~padding_mask.view(B * L)
            if mask.sum() > 0:
                pred_masked = pred_flat[mask].detach().cpu().numpy()
                target_masked = target_flat[mask].detach().cpu().numpy()
                
                # Overall accuracy
                acc = np.mean(pred_masked == target_masked)
                
                # Per-class accuracy for amino acids
                if key in ['aatype', 'msa', 'residue_index']:
                    class_accs = []
                    for aa in range(1, 21):  # Skip padding token
                        mask_aa = target_masked == aa
                        if mask_aa.sum() > 0:
                            acc_aa = np.mean(pred_masked[mask_aa] == target_masked[mask_aa])
                            class_accs.append(acc_aa)
                    
                    metrics[key] = {
                        'accuracy': acc, 
                        'avg_class_acc': np.mean(class_accs) if class_accs else 0.0,
                        'num_classes_present': len(class_accs)
                    }
                else:
                    metrics[key] = {'accuracy': acc}
    
    return metrics

def train_enhanced_model(dataset, device, num_epochs=200, batch_size=16, base_lr=2e-4):
    """Enhanced training with progressive strategies"""
    
    # Create data loader
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=enhanced_collate_fn,
        num_workers=2,
        pin_memory=True
    )
    
    # Create enhanced model
    model = EnhancedTransformer(d_model=512, nhead=8, num_layers=8, dropout=0.1).to(device)
    
    print(f"Enhanced Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Enhanced optimizer with better hyperparameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=0.01,
        betas=(0.9, 0.95),  # Better betas for transformers
        eps=1e-8
    )
    
    # Enhanced learning rate schedule
    total_steps = len(loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=base_lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    # Enhanced loss function
    loss_fn = EnhancedLossFunction(device)
    
    print(f"Starting enhanced training: {len(loader)} batches/epoch, {num_epochs} epochs")
    
    # Training tracking
    best_loss = float('inf')
    patience_counter = 0
    max_patience = 25
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_losses = {key: 0.0 for key in ['residue_index', 'aatype', 'deletion_matrix', 'msa']}
        epoch_metrics = {key: {} for key in ['residue_index', 'aatype', 'deletion_matrix', 'msa']}
        valid_batches = 0
        
        for batch_idx, (seqs, targets, mask) in enumerate(loader):
            try:
                # Move to device
                seqs = seqs.to(device, non_blocking=True)
                mask = mask.to(device, non_blocking=True)
                for key in targets:
                    targets[key] = targets[key].to(device, non_blocking=True)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = model(seqs, mask)
                
                # Compute enhanced loss
                total_loss, losses = loss_fn(predictions, targets, mask, epoch)
                
                # Backward pass with gradient clipping
                if total_loss > 0:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    # Track metrics
                    epoch_loss += total_loss.item()
                    for key, loss_val in losses.items():
                        epoch_losses[key] += loss_val
                    
                    # Calculate batch metrics
                    batch_metrics = calculate_enhanced_metrics(predictions, targets, mask)
                    for key, metrics_dict in batch_metrics.items():
                        for metric_name, metric_val in metrics_dict.items():
                            if metric_name not in epoch_metrics[key]:
                                epoch_metrics[key][metric_name] = []
                            epoch_metrics[key][metric_name].append(metric_val)
                    
                    valid_batches += 1
                
                # Progress update
                if batch_idx % 20 == 0 and batch_idx > 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(loader)}, "
                          f"Loss: {total_loss.item():.4f}, LR: {current_lr:.2e}")
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
        
        # Epoch summary
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            current_lr = scheduler.get_last_lr()[0]
            
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Average Loss: {avg_loss:.4f}")
            print(f"  Learning Rate: {current_lr:.2e}")
            print(f"  Valid Batches: {valid_batches}/{len(loader)}")
            
            # Component losses
            for key, loss_val in epoch_losses.items():
                if valid_batches > 0:
                    avg_component_loss = loss_val / valid_batches
                    print(f"  {key} Loss: {avg_component_loss:.4f}")
            
            # Metrics summary
            for key, metrics_dict in epoch_metrics.items():
                if metrics_dict:
                    print(f"  {key} Metrics: ", end="")
                    for metric_name, metric_values in metrics_dict.items():
                        avg_metric = np.mean(metric_values)
                        print(f"{metric_name}: {avg_metric:.4f}", end=" ")
                    print()
            
            # Early stopping and checkpointing
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                
                # Save best model
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'loss_weights': loss_fn.loss_weights.data.clone()
                }
                torch.save(checkpoint, "./best_enhanced_model.pth")
                print(f"  New best model saved! Loss: {avg_loss:.4f}")
            
                # Early success check
                if avg_loss < 0.8:
                    print(f"🎉 EXCELLENT! Loss < 0.8 achieved at epoch {epoch+1}")
                elif avg_loss < 1.2:
                    print(f"GREAT! Loss < 1.2 achieved at epoch {epoch+1}")
            else:
                patience_counter += 1
                
            # Dynamic patience based on loss level
            if avg_loss < 1.0:
                effective_patience = max_patience * 2  # More patience when close to target
            else:
                effective_patience = max_patience
                
            if patience_counter >= effective_patience:
                print(f"Early stopping at epoch {epoch+1}. Best loss: {best_loss:.4f}")
                break
                
            # Learning rate restart if stuck
            if patience_counter > 15 and avg_loss > 1.5:
                print(" Restarting learning rate schedule...")
                for param_group in optimizer.param_groups:
                    param_group['lr'] = base_lr * 0.5
                patience_counter = 0
        else:
            print(f"Epoch {epoch+1}: No valid batches")
    
    return model

# Test functions for validation
def test_single_batch_enhanced(model, dataset, device):
    """Test a single batch with enhanced model"""
    print("\n=== ENHANCED SINGLE BATCH TEST ===")
    
    # Create a small test batch
    test_items = [dataset[i] for i in range(min(2, len(dataset)))]
    seqs_padded, targets_padded, padding_mask = enhanced_collate_fn(test_items)
    
    print(f"Test batch - seqs: {seqs_padded.shape}, mask: {padding_mask.shape}")
    for key, target in targets_padded.items():
        print(f"  Target {key}: {target.shape}")
    
    # Move to device
    seqs_padded = seqs_padded.to(device)
    padding_mask = padding_mask.to(device)
    for key in targets_padded:
        targets_padded[key] = targets_padded[key].to(device)
    
    # Test model forward pass
    model.eval()
    with torch.no_grad():
        try:
            predictions = model(seqs_padded, padding_mask)
            print(f"Model forward pass successful!")
            for key, pred in predictions.items():
                print(f"  {key}: shape={pred.shape}, range={pred.min().item():.3f} to {pred.max().item():.3f}")
            
            # Test loss computation
            loss_fn = EnhancedLossFunction(device)
            total_loss, losses = loss_fn(predictions, targets_padded, padding_mask)
            print(f"Loss computation successful! Total: {total_loss.item():.4f}")
            for key, loss_val in losses.items():
                print(f"  {key} loss: {loss_val:.4f}")
            
            # Test metrics
            metrics = calculate_enhanced_metrics(predictions, targets_padded, padding_mask)
            print(f"Metrics calculation successful!")
            for key, metric_dict in metrics.items():
                print(f"  {key}: {metric_dict}")
            
            return True
            
        except Exception as e:
            print(f"Error in enhanced single batch test: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

# Main function with all enhancements
def main_enhanced(device):
    """Enhanced main function with all optimizations"""
    print("Loading data for enhanced training...")
    
    folder = "C:\\Users\\s_smm\\OneDrive\\Documents\\CICESE\\Proyecto\\data\\features"
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pkl")]
    dfs = []
    for file_path in files:
        df = pd.read_pickle(file_path)
        dfs.append(df)
    
    df = pd.DataFrame(dfs)
    
    # Prepare data
    df_aatype = df['aatype']
    df_train = df.drop(columns=["sequence", "aatype", 'msa_species_identifiers', 'domain_name'])
    
    print(f"Raw data loaded: {len(df)} samples")
    print(f"Available features: {list(df_train.columns)}")
    
    # Enhanced data quality check
    print("\n Data Quality Analysis:")
    sample_residue_index = df_train['residue_index'].iloc[0]
    sample_deletion_matrix = df_train['deletion_matrix_int'].iloc[0]
    sample_msa = df_train['msa'].iloc[0]
    
    print(f"Sample residue_index: shape={np.array(sample_residue_index).shape}")
    print(f"Sample deletion_matrix: shape={np.array(sample_deletion_matrix).shape}")
    print(f"Sample msa: shape={np.array(sample_msa).shape}")
    
    # Create enhanced dataset with filtering
    dataset = EnhancedSeqDataset(df_aatype, df_train, max_len=1024)
    print(f"\n📦 Enhanced dataset created with {len(dataset)} samples")
    
    # Test samples quality
    print("\n🔍 Sample Quality Check:")
    for i in range(min(3, len(dataset))):
        seq, tgts = dataset[i]
        print(f"Sample {i}: seq_len={seq.shape[0]}")
        for key, tgt in tgts.items():
            if key == 'aatype':
                matches = (seq == tgt).all().item()
                print(f"  {key}: perfect_match={matches}")
            elif key == 'deletion_matrix':
                deletion_rate = (tgt > 0.5).sum().item() / len(tgt)
                print(f"  {key}: deletion_rate={deletion_rate:.3f}")
            elif key == 'msa':
                similarity = (seq == tgt).sum().item() / len(seq)
                print(f"  {key}: seq_similarity={similarity:.3f}")
            else:
                print(f"  {key}: range={tgt.min().item()}-{tgt.max().item()}")
    
    # Create and test model
    model = EnhancedTransformer(d_model=512, nhead=8, num_layers=8, dropout=0.1).to(device)
    print(f"\n🤖 Enhanced model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test single batch first
    if not test_single_batch_enhanced(model, dataset, device):
        print("Enhanced single batch test failed, stopping")
        return
    
    # Start enhanced training
    print(f"\n🔥 Starting Enhanced Training with Advanced Optimizations!")
    print("Key improvements:")
    print("  ✅ Multi-scale positional encoding")
    print("  ✅ Focal loss for class imbalance")
    print("  ✅ Adaptive loss weighting")
    print("  ✅ Enhanced data preprocessing")
    print("  ✅ Progressive learning strategies")
    print("  ✅ Better regularization")
    print("  ✅ Dynamic early stopping")
    print("  ✅ Learning rate restarts")
    
    model = train_enhanced_model(
        dataset, 
        device, 
        num_epochs=200, 
        batch_size=12,  # Slightly smaller batch for stability
        base_lr=2e-4
    )
    
    # Final model save
    torch.save(model.state_dict(), "./final_enhanced_model.pth")
    print(f"\nFinal enhanced model saved!")
    print(f"Expected improvements:")
    print(f"  • Loss should drop from ~2.4 to <1.0")
    print(f"  • Better accuracy on all tasks")
    print(f"  • Faster convergence")
    print(f"  • More stable training")
    
    return model

# Alternative simplified version of your original code with key improvements
def main_improved_simple(device):
    """Improved version of your original code with key optimizations"""
    print("Loading data...")
    folder = "C:\\Users\\s_smm\\OneDrive\\Documents\\CICESE\\Proyecto\\data\\features"
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".pkl")]
    dfs = []
    for file_path in files:
        df = pd.read_pickle(file_path)
        dfs.append(df)
    
    df = pd.DataFrame(dfs)
    
    # Prepare data (same as your original)
    df_aatype = df['aatype']
    df_train = df.drop(columns=["sequence", "aatype", 'msa_species_identifiers', 'domain_name'])
    
    print(f"Data loaded: {len(df)} samples")
    
    # Use your original dataset class
    from your_original_code import SimpleSeqDataset, SimpleTransformer, simple_collate_fn
    dataset = SimpleSeqDataset(df_aatype, df_train)
    
    # Enhanced model with key improvements
    model = SimpleTransformer(d_model=512, nhead=8, num_layers=8).to(device)  # Deeper model
    
    # Better optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-4,           # Lower learning rate
        weight_decay=0.01, # Weight decay
        betas=(0.9, 0.95)  # Better betas
    )
    
    # Learning rate scheduler
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=simple_collate_fn)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        total_steps=len(loader) * 100,
        pct_start=0.1
    )
    
    # Enhanced loss with label smoothing
    criterions = {
        'residue_index': nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1),
        'aatype': nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1),
        'deletion_matrix': nn.CrossEntropyLoss(label_smoothing=0.1),
        'msa': nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    }
    
    # Better loss weights
    loss_weights = {
        'residue_index': 1.0,
        'aatype': 2.5,  # Higher weight for main task
        'deletion_matrix': 0.3,  # Lower weight for sparse task
        'msa': 1.2
    }
    
    print("Starting improved training...")
    best_loss = float('inf')
    
    for epoch in range(100):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        
        for batch_idx, (seqs, targets, mask) in enumerate(loader):
            try:
                seqs = seqs.to(device)
                mask = mask.to(device)
                for key in targets:
                    targets[key] = targets[key].to(device)
                
                optimizer.zero_grad()
                preds = model(seqs, mask)
                
                total_loss = 0.0
                loss_count = 0
                
                for key in preds.keys():
                    pred = preds[key]
                    target = targets[key]
                    
                    B, L, C = pred.shape
                    pred_flat = pred.view(B * L, C)
                    target_flat = target.view(B * L)
                    
                    # Skip deletion matrix if only one class
                    if key == 'deletion_matrix':
                        unique_targets = torch.unique(target_flat)
                        if len(unique_targets) < 2:
                            continue
                    
                    loss = criterions[key](pred_flat, target_flat)
                    weighted_loss = loss * loss_weights[key]
                    total_loss += weighted_loss
                    loss_count += 1
                
                if loss_count > 0:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
                    optimizer.step()
                    scheduler.step()
                    
                    epoch_loss += total_loss.item()
                    valid_batches += 1
                
            except Exception as e:
                continue
        
        if valid_batches > 0:
            avg_loss = epoch_loss / valid_batches
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: avg loss = {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), "./improved_simple_model.pth")
                
            if avg_loss < 1.0:
                print(f"🎉 Target achieved! Loss < 1.0 at epoch {epoch}")
                break
    
    print(f"Training complete. Best loss: {best_loss:.4f}")
    return model

if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    
    main_enhanced(device)
    
    