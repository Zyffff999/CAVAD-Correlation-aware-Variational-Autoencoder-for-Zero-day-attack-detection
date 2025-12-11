import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple, List
import os
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

class VAETrainer:
    """
    Optimized VAE Trainer.
    Focus: Clear logging of decomposed losses (Total, Recon, KL) and robust anomaly scoring.
    """
    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            benign_label: int = 0,
            output_dir: str = "./outputs"
    ):
        self.model = model.to(device)
        self.device = device
        self.benign_label = benign_label
        self.output_dir = output_dir
        
        # Simple history tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_recon': [],
            'val_kl': [],
            'auc': [],  # <--- Was missing
            'active_dims': [],  # <--- Was missing
            'condition_number': []  # <--- Was missing
        }
    # ================================================================
    # TRAINING LOOP
    # ================================================================
    def train_epoch(
            self,
            dataloader: DataLoader,
            optimizer: torch.optim.Optimizer,
            kld_weight: float = 1.0,
            free_bits: float = 0.0,
            clip_grad: Optional[float] = 1.0,
            use_amp: bool = False,
            scaler: Optional[object] = None,
            **kwargs # Accepts extra args like correlation_penalty_weight if passed
    ) -> Dict[str, float]:
        
        self.model.train()
        
        # Accumulators for logging
        acc_total = 0.0
        acc_recon = 0.0
        acc_kld = 0.0
        count = 0

        # Progress bar with dynamic description
        pbar = tqdm(dataloader, desc="Train", leave=False, dynamic_ncols=True)
        
        for data, _ in pbar:
            data = data.to(self.device).float()
            optimizer.zero_grad(set_to_none=True)
            
            # 1. Forward & Loss Calculation
            # We pass **kwargs so you can handle specific penalties in vae.py if needed
            if use_amp and scaler:
                with torch.cuda.amp.autocast():
                    z, mu, params, recon = self.model(data)
                    loss, r_loss, k_loss, _ = self.model.vae_loss_full(
                        recon, data, mu, params, 
                        kld_weight=kld_weight, 
                        free_bits=free_bits,
                        **kwargs 
                    )
                scaler.scale(loss).backward()
                if clip_grad:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                scaler.step(optimizer)
                scaler.update()
            else:
                z, mu, params, recon = self.model(data)
                loss, r_loss, k_loss, _ = self.model.vae_loss_full(
                    recon, data, mu, params, 
                    kld_weight=kld_weight, 
                    free_bits=free_bits,
                    **kwargs
                )
                loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                optimizer.step()

            # 2. Logging Accumulation
            bs = data.size(0)
            acc_total += loss.item() * bs
            acc_recon += r_loss.item() * bs
            acc_kld += k_loss.item() * bs
            count += bs

            # 3. Real-time Progress Bar Update
            pbar.set_postfix({
                "Tot": f"{loss.item():.2f}",
                "Rec": f"{r_loss.item():.2f}",
                "KL": f"{k_loss.item():.2f}"
            })

        # Epoch Averages
        avg_total = acc_total / count
        avg_recon = acc_recon / count
        avg_kld = acc_kld / count
        
        self.history['train_loss'].append(avg_total)
        
        return {
            'total_loss': avg_total,
            'recon_loss': avg_recon,
            'kld_loss': avg_kld
        }

    # ================================================================
    # EVALUATION LOOP
    # ================================================================
    def evaluate(
            self,
            dataloader: DataLoader,
            kld_weight: float = 1.0,
            free_bits: float = 0.0,
            **kwargs
    ) -> Dict[str, float]:
        
        self.model.eval()
        acc_total = 0.0
        acc_recon = 0.0
        acc_kld = 0.0
        count = 0

        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device).float()
                
                z, mu, params, recon = self.model(data)
                loss, r_loss, k_loss, _ = self.model.vae_loss_full(
                    recon, data, mu, params,
                    kld_weight=kld_weight,
                    free_bits=free_bits,
                    **kwargs
                )
                
                bs = data.size(0)
                acc_total += loss.item() * bs
                acc_recon += r_loss.item() * bs
                acc_kld += k_loss.item() * bs
                count += bs

        if count == 0: 
            return {'val_loss': float('inf'), 'v_recon': float('inf'), 'v_kld': float('inf')}

        avg_total = acc_total / count
        avg_recon = acc_recon / count
        avg_kld = acc_kld / count

        self.history['val_loss'].append(avg_total)
        self.history['val_recon'].append(avg_recon)
        self.history['val_kl'].append(avg_kld)

        return {
            'val_loss': avg_total,
            'v_recon': avg_recon,
            'v_kld': avg_kld
        }

    # ================================================================
    # ANOMALY SCORING & STATISTICS (Required by Test.py)
    # ================================================================
    def compute_anomaly_scores(self,
                               dataloader: DataLoader,
                               method: str = 'combined',
                               reference_stats: Dict[str, torch.Tensor] = None) -> List[float]:
        """
        Computes anomaly scores for the entire dataloader.
        Used by test.py for evaluation.
        """
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device).float()
                z, mu, params, recon = self.model(data)

                # 1. Reconstruction Error (MSE)
                recon_flat = recon.reshape(data.shape[0], -1)
                x_flat = data.reshape(data.shape[0], -1)
                recon_loss = F.mse_loss(recon_flat, x_flat, reduction='none').mean(dim=1)

                score = recon_loss # Default

                # 2. Mahalanobis / Combined
                if method in ['mahalanobis', 'combined', 'whitened_l2','kl','combined-1']:
                    if reference_stats is None:
                        raise ValueError(f"Method {method} requires reference_stats")
                        
                    mu_ref = reference_stats['reference_mu'].to(self.device)
                    cov_inv_ref = reference_stats['reference_cov_inv'].to(self.device)
                    
                    mah_dist = self.model.compute_mahalanobis_distance(data, mu_ref, cov_inv_ref)
                    kl_per_sample = self.model.compute_kl_divergence(mu, params)
                    if method == 'mahalanobis':
                        score = mah_dist
                    elif method == 'combined':
                        score = 0.01 * recon_loss + mah_dist
                    elif method == 'whitened_l2':
                        invsqrt = reference_stats['reference_cov_invsqrt'].to(self.device)
                        mu_tilde = (mu - mu_ref) @ invsqrt.T
                        score = mu_tilde.norm(dim=1)
                    elif method == 'kl':
                        score = kl_per_sample
                    elif method == 'combined-1':
                        score = 0.01*recon_loss + kl_per_sample

                scores.extend(score.cpu().numpy().tolist())

        return scores

    def compute_reference_statistics(self, dataloader: DataLoader) -> Dict[str, torch.Tensor]:
        """
        Robust statistic estimation (OAS Shrinkage) for Mahalanobis distance.
        """
        self.model.eval()
        all_mu = []
        
        # 1. Collect Latent Means
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device).float()
                _, mu = self.model.get_features(data)
                all_mu.append(mu.cpu())
        
        all_mu = torch.cat(all_mu, dim=0).double()
        N, D = all_mu.shape
        
        # 2. Compute Mean
        mean = all_mu.mean(dim=0)
        X = all_mu - mean
        
        # 3. Compute Covariance with OAS Shrinkage (Numerically Stable)
        sample_cov = (X.T @ X) / (N - 1)
        trace = torch.trace(sample_cov)
        trace2 = torch.trace(sample_cov @ sample_cov)
        
        mu_trace = trace / D
        alpha = mu_trace
        num = (1 - 2/D) * trace2 + trace**2
        den = (N + 1 - 2/D) * (trace2 - trace**2/D)
        shrinkage = 1.0 if den == 0 else torch.clamp(num / den, 0.0, 1.0)
        
        shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * alpha * torch.eye(D)
        
        # 4. Compute Inverses
        # Use Cholesky for stability: Cov = L L^T => Cov^-1 = L^-T L^-1
        jitter = 1e-6 * torch.eye(D)
        try:
            L = torch.linalg.cholesky(shrunk_cov + jitter)
            # Calculate inverse using triangular solve
            # Solve L * Y = I -> Y = L^-1
            L_inv = torch.linalg.solve_triangular(L, torch.eye(D, dtype=torch.double), upper=False)
            cov_inv = L_inv.T @ L_inv
            
            # For whitened L2 (Inverse Sqrt)
            # SVD is safer for sqrt
            U, S, Vt = torch.linalg.svd(shrunk_cov)
            cov_invsqrt = U @ torch.diag(S.rsqrt()) @ Vt
            
        except:
            print("Warning: Cholesky failed, falling back to pinv")
            cov_inv = torch.linalg.pinv(shrunk_cov)
            cov_invsqrt = torch.linalg.pinv(shrunk_cov).sqrt() # Approximate

        return {
            'reference_mu': mean.float(),
            'reference_cov': shrunk_cov.float(),
            'reference_cov_inv': cov_inv.float(),
            'reference_cov_invsqrt': cov_invsqrt.float()
        }
    
    def save_checkpoint(self, filepath, epoch, optimizer=None):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
            'history': self.history
        }, filepath)

    def _quick_auc(self, loader, ref_stats):
        # Helper to get AUC quickly
        scores, labels = [], []
        mu_ref = ref_stats['reference_mu'].to(self.device)
        cov_inv_ref = ref_stats['reference_cov_inv'].to(self.device)

        with torch.no_grad():
            for data, y in loader:
                data = data.to(self.device).float()
                dist = self.model.compute_mahalanobis_distance(data, mu_ref, cov_inv_ref)
                scores.append(dist.cpu().numpy())
                labels.append(y.numpy())

        y_true = np.concatenate(labels) != self.benign_label
        y_score = np.concatenate(scores)
        if len(np.unique(y_true)) < 2: return 0.5
        return roc_auc_score(y_true, y_score)

    def run_diagnostics(self, train_loader: DataLoader, test_loader: DataLoader, epoch: int):
        """
        Runs a quick health check and anomaly detection test.
        Prints a single line summary unless issues are found.
        """
        if epoch % 15 != 0: return
        self.model.eval()
        try:
            batch = next(iter(train_loader))[0].to(self.device).float()
            with torch.no_grad():
                _, mu, params, _ = self.model(batch)

            # Quick Stats
            active_dims = (mu.std(0) > 0.1).sum().item()

            # Condition Number (if correlated)
            cond_num = 0.0
            if self.model.training_mode == 'correlated':
                L = params
                Sigma = torch.bmm(L, L.transpose(-1, -2))
                evals = torch.linalg.eigvalsh(Sigma)
                cond_num = (evals[:, -1] / (evals[:, 0] + 1e-6)).mean().item()

            # 2. Anomaly Detection Performance
            # We use the robust reference stats calculator
            ref_stats = self.compute_reference_statistics(train_loader)

            auc = (self._quick_auc

                   (test_loader, ref_stats))

            self.history['auc'].append(auc)
            self.history['active_dims'].append(active_dims)
            self.history['condition_number'].append(cond_num)

            # 3. Print Clean Summary
            print(f"\n[Epoch {epoch} Diagnostics] "
                  f"AUC: {auc:.4f} | "
                  f"Active Dims: {active_dims}/{self.model.latent_dim} | "
                  f"Cond #: {cond_num:.1f}")

            if active_dims < 2:
                print("⚠️  WARNING: Posterior Collapse Detected!")
            if cond_num > 1000:
                print("⚠️  WARNING: Latent Space Ill-Conditioned!")

        except Exception as e:
            print(f"[Diagnostics Error] {e}")